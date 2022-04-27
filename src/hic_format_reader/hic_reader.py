import numpy as np
import os,struct
import random
import straw
from scipy.sparse import csr_matrix, coo_matrix, vstack, hstack
from scipy import sparse

def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')
        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b

def read_hic_header(hicfile):
    if not os.path.exists(hicfile):
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    magic_string = struct.unpack('<3s', req.read(3))[0]
    req.read(1)
    if (magic_string != b"HIC"):
        return None  # this is not a valid .hic file

    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)

    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)

    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length

    info['chromsizes'] = chromsizes

    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)

    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info


def matrix_extract(chrN1, chrN2, binsize, hicfile, normalization='KR'):
    result, norms = straw.straw(normalization, hicfile, str(chrN1), str(chrN2), 'BP', binsize)
        
    compact_idx_norm_x = list(np.where(np.isnan(norms[0])^True)[0])
    compact_idx_norm_y = list(np.where(np.isnan(norms[1])^True)[0])
    
    

    compact_idx = []
    if compact_idx_norm_x == compact_idx_norm_y:
        compact_idx = compact_idx_norm_x
    else:
        raise Exception("Invalid Normalization vector")
    row = [r//binsize for r in result[0]]
    col = [c//binsize for c in result[1]]
    
    value = result[2]
    Nrow = max(row) + 1
    Ncol = max(col) + 1
    
    chr_total_size = read_hic_header(hicfile)['chromsizes'][str(chrN1)]
    
    #N = max(Nrow, Ncol)
    N = chr_total_size//binsize + 1

    mat = csr_matrix((value, (row,col)), shape=(N,N))
    mat = csr_matrix.todense(mat)
    mat = mat.T
    mat = mat + np.tril(mat, -1).T

    mat = mat.astype(int)
    
    return mat, compact_idx