#!/usr/bin/env python
import sys
import os
import re

SQL_query_QSO = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'QSO') AND (zWarning = 0)"""

SQL_query_GAL = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'GALAXY') AND (zWarning = 0)"""

SQL_query_STAR = """SELECT TOP {0}
s.specObjID, s.class, s.z as redshift, s.zErr as redshiftError, s.spectroSynFluxIvar_u, s.spectroSynFluxIvar_g, s.spectroSynFluxIvar_r, s.spectroSynFluxIvar_z, s.spectroSynFluxIvar_i, s.spectroSynFlux_i, s.spectroSynFlux_z, s.spectroSynFlux_u, s.spectroSynFlux_g, s.plate, s.mjd, s.fiberid, s.run2d, p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z, p.fieldID, p.err_u, p.err_g, p.err_r, p.err_i, p.err_z, p.petroRad_u, p.petroRad_g, p.petroRad_r, p.petroRad_i, p.petroRad_z, p.petroRadErr_u, p.petroRadErr_g, p.petroRadErr_r, p.petroRadErr_z, p.petroRadErr_i
FROM PhotoObj AS p
   JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
   (s.class = 'STAR') AND (zWarning = 0)"""

def getObjTable(SQL_query, format="csv", fname="queryResults", TableName="",N=100000):
    """Retrieve table of N SDSS objects using SQL query
    params: SQL_query [str] - SQL query to be executed
            format [str] - format of table, options are 'csv', 'html', 'xml', 'json'
            TableName [str] - name of table, default is empty string
            N [int] - number of objects to retrieve, default is 100000, limit is 500000
    returns: query [str] - url of reformatted query, downloaded with wget and saved as {fname}.{format}
    """
    SQL_query = SQL_query.format(N)
    base_url = "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd="
    SQL_formatted = re.sub(r"\n", "%0D%0A", SQL_query)
    SQL_formatted = re.sub(r"\+", "%2B", SQL_formatted)
    SQL_formatted = re.sub(r"\s", "+", SQL_formatted)
    SQL_formatted = re.sub(r"=", "%3D", SQL_formatted)
    SQL_formatted = re.sub(r",", "%2C", SQL_formatted)
    SQL_formatted = re.sub(r"'", "%27", SQL_formatted)
    SQL_formatted = re.sub(r"\(", "%28", SQL_formatted)
    SQL_formatted = re.sub(r"\)", "%29", SQL_formatted)
    query = base_url + SQL_formatted + "%0D%0A%0D%0A&format={0}&TableName={1}".format(format, TableName)
    os.system("wget -O {0}.{1} '".format(fname,format) + query + "'")
    return query

def getSpectrum(id,format='png',spec='lite'):
    """Retrieve spectra of SDSS objects
    params: id [dict] - dictionary of object identifiers, requires at least 'specObjID'
            format [str] - format of spectrum, options are 'png', 'fits', 'csv'
            spec [str] - either 'lite' (default) or 'full'
    returns: url [str] - url of spectrum, downloaded with wget and saved as spectra/spectrum_{specObjID}.{format}
    """

    if format == 'png':
        base_url = "https://skyserver.sdss.org/dr18/en/get/specById.ashx?ID="
        url = base_url + str(int(id['specObjID']))
    elif format == 'fits' or format == 'csv':
        url = "http://dr18.sdss.org/optical/spectrum/view/data/format={4}/spec={5}?plateid={0}&mjd={1}&fiberid={2}&reduction2d={3}".format(str(int(id['plate'])), str(int(id['mjd'])), str(int(id['fiberid'])), str(int(id['run2d'])),format,spec)
    os.system("wget -O spectra/spectrum_{0}.{1} '".format(str(int(id['specObjID'])),format) + url + "'")
    return url

def getImg(id,opt="GL",width=512,height=512,scale=0.4):
    """Retrieve images of SDSS objects
    params: id [dict] - dictionary of object identifiers, requires at least 'specObjID', 'ra', and 'dec'
            opt [str] - options for image cutout, documentation: https://skyserver.sdss.org/dr14/en/help/docs/api.aspx#imgcutout; defaults to 'GL' (grid + label)
            width [int] - width of image in pixels
            height [int] - height of image in pixels
            scale [float] - scale of image in arcsec/pixel
    
    returns: url [str] - url of image cutout, downloaded with wget and saved as images/image_{specObjID}.jpeg 
    """

    url = "http://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?ra={0}&dec={1}&width={2}&height={3}&scale={4}&opt={5}".format(id['ra'],id['dec'],width,height,scale,opt)
    os.system("wget -O images/image_{0}.jpeg '".format(str(int(id['specObjID']))) + url + "'")
    return url

def getID(row,fname):
    row += 2 # skip table name + column names
    with open(fname,"r") as f:
        lines = f.readlines()
        d = lines[row].split(",")
        colNames = lines[1].split(",")
    return dict(zip(colNames,d))






