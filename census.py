"""
To use this module we need to download some data

TODO Automate

download:
    ks102ew_2011_oa.zip
    Output_Area_December_2011_Generalised_Clipped_Boundaries_in_England_and_Wales.zip
    http://webarchive.nationalarchives.gov.uk/20160110200239/http://www.ons.gov.uk/ons/external-links/social-media/g-m/2011-oas-to-2011-lower-layer-super-output-areas--lsoas---middle-layer-super-output-areas--msoa--and-lads.html

unzip, e.g. using:
    import zipfile
    zipref = zipfile.ZipFile('Output_Area_December_2011_Generalised_Clipped_Boundaries_in_England_and_Wales.zip')
    zipref.extractall("oaboundaries")
    zipref = zipfile.ZipFile('ks102ew_2011_oa.zip')
    zipref.extractall("oadata")
    zipref = zipfile.ZipFile('Output_areas_to_LSOA.zip')
    zipref.extractall('oa_to_lsoa')
"""


import pandas as pd
import shapefile
import matplotlib.pyplot as plt
import numpy as np
#import tripy
from earclip import triangulate
import pickle

def earclip(a):
    m = np.mean(a,0)
    return triangulate(a.tolist())
    #return tripy.earclip(a-m)+m


def get_census_data(shapefilename, oafilename, oadescfilename, oadatafilename,box=None,refresh=False, verbose=False):        
    """
        newX, newY, newtestX, testY = get_census_data(box=None)
    
    The test data is the Output Areas (OAs) as inputs (each row is one OA),
    with outputs the population count in each OA.
    The training data is the MSOA (larger areas which consist of several OAs)
    the outputs are again the aggregate count in each MSOA.
    Note that here we only include those OAs that are within the box specified,
    so some MSOAs may be incomplete (but the count values will be for the subset
    of OAs included).
    
    shapefile, oafile, oadesc, oadata = locations of files
    
    box=[East, West, South, North], e.g. [328e3,335e3,185e3,192e3]
    
    returns
        Training and test pairs of inputs and outputs
        
    Example
        import census
        X,Y,testX,testY = census.get_census_data(shapefilename='oaboundaries/Output_Area_December_2011_Generalised_Clipped_Boundaries_in_England_and_Wales.shp',
    oafilename='oa_to_lsoa/OA11_LSOA11_MSOA11_LAD11_EW_LUv2.csv', oadescfilename='oadata/KS102ew_2011_oa/KS102EW_2011STATH_NAT_OA_REL_1.4.4/KS102EWDESC0.CSV', oadatafilename='oadata/KS102ew_2011_oa/KS102EWDATA.CSV',box=[328e3,335e3,185e3,192e3])
    """
    
    if box is not None:
        cachename = 'census_%d_%d_%d_%d.p' % (box[0],box[1],box[2],box[3])
    else:
        cachename = 'census.p'
        
    if not refresh:
        try:
            if verbose: print("Trying cache...")
            newX, Y, newtestX, testY = pickle.load(open(cachename,'rb'))
            if verbose: print("Cache loaded.")
            if verbose:
                print("Loaded from cache, %d training polygons and %d testing polygons" % (len(Y),len(testY)))
            return newX, Y, newtestX, testY
        except FileNotFoundError:
            pass
        if verbose: print("cache not found.");
    else:
        if verbose: print("refreshing cache.");

    
    
    
    #Load shapes and dataframes
    sf = shapefile.Reader(shapefilename)
    dfoas = pd.read_csv(oafilename,encoding='latin-1')
    dfdesc = pd.read_csv(oadescfilename)
    dfdata = pd.read_csv(oadatafilename) #TODO Allow different sorts of data

    #Combine the variable info into one column name,
    #e.g. 'Age 25 to 29 Count' or 'Age 25 to 29 Percentage'
    cols = dfdata.columns.values
    for code, desc, unit in zip(dfdesc['ColumnVariableCode'].values,dfdesc['ColumnVariableDescription'].values,dfdesc['ColumnVariableMeasurementUnit'].values):
        cols[cols==code]=desc+' '+unit    
    dfdata.columns = cols

    #Iterate through the OA shapes, record which MSOA each belongs to, get a list
    #of population counts in each and get a list of the geometries of the OAs.
    msoacounts = {}
    msoapatches = {}
    testX = []
    testY = []
    for shaperec in sf.iterShapeRecords():
        if box is not None:
            keep = (shaperec.shape.bbox[2]>box[0]) & (shaperec.shape.bbox[0]<box[1]) & (shaperec.shape.bbox[3]>box[2]) & (shaperec.shape.bbox[1]<box[3])
            if not keep:
                continue
        
        #get the OA code and find the OA dataframe row, and grab the total number of people in the OA.
        dfrow = dfdata[dfdata['GeographyCode']==shaperec.record[1]]
        count = dfrow['All categories: Age Count'].values[0]
        testY.append(count)
        
        #get the MSOA the OA belongs to
        msoa = dfoas[dfoas['OA11CD']==shaperec.record[1]].MSOA11CD.values[0]
        if msoa not in msoacounts:
            msoacounts[msoa] = 0
            msoapatches[msoa] = []
        
        #track the count for each MSOA
        msoacounts[msoa]+=count

        #get the shape's boundary points
        ps = np.array(shaperec.shape.points)
        #convert to km
        ps = ps / 1000.0
        
        #convert to a bunch of triangles
        tri = earclip(ps)
        flatpoly = np.array(tri)
        #flatten and add as a test data row, and into the msoa patches
        testX.append(np.reshape(flatpoly,[1,np.prod(flatpoly.shape)]))
        msoapatches[msoa].append(ps)
        #if shaperec.record[0]==38620:
        #    temp = {'fp':flatpoly,'ps':ps}
        
    #build the training data (this is the aggregated MSOAs made from OAs)
    X = []
    Y = []
    #for each MSOA...
    for msoa in msoapatches:
        flatpolys = np.zeros([1,0])
        #loop through each OA making up each MSOA;
        for ps in msoapatches[msoa]:
            #convert to triangles and add to the flatpolys list
            tri = earclip(ps)
            flatpoly = np.array(tri)
            flatpoly = np.reshape(flatpoly,[1,np.prod(flatpoly.shape)])
            flatpolys = np.c_[flatpolys,flatpoly]
        #flatten the list of flatpolys
        X.append(np.reshape(flatpolys,[1,np.prod(flatpolys.shape)]))
        Y.append(msoacounts[msoa])
    Y = np.array(Y)[:,None]
    testY = np.array(testY)[:,None]    
    #X and Y contain the training data.
    #testX and testY contain the test data.

    #need to build a X to make it a proper matrix
    #(i.e. each row should have same length)
    testmaxlen = max([x.shape[1] for x in testX])
    trainmaxlen = max([x.shape[1] for x in X])
    maxlen = max([testmaxlen,trainmaxlen])
    newX = np.zeros([len(X),maxlen])*np.nan
    for i,x in enumerate(X):
        newX[i:i+1,0:x.shape[1]] = x
        
    #similarly we need to make the test data into a matrix
    newtestX = np.zeros([len(testX),maxlen])*np.nan
    for i,x in enumerate(testX):
        newtestX[i:i+1,0:x.shape[1]] = x
    
    if verbose:
        print("Created %d training polygons and %d testing polygons" % (len(Y),len(testY)))

    pickle.dump((newX, Y, newtestX, testY),open(cachename,'wb'))    
    return newX, Y, newtestX, testY#, temp
    
def getpred(m,positions):
    """
    predictions = getpred(m,positions)
    
    m = model (with shapekernel)
    positions = numpy array Nx2 of locations
    
    Uses the shape kernel to make a prediction of population density at a
    given position.
    """
    nd = m.X.shape[1]
    simpleX = np.zeros([len(positions),nd])*np.nan
    simpleX[:,0:12] = np.array([0,0,0,1,1,0,1,1,0,1,1,0])*0.001 #square metre
    simpleX[:,0:12:2] += np.repeat(positions[:,0:1],6,1)
    simpleX[:,1:12:2] += np.repeat(positions[:,1:2],6,1)
    p, _ = m.predict(simpleX)
    return p    
