import numpy as np
import sys
import os

def writeData(data, outputFilename):
    """
    Writes data to a tiff, hdf5, or npy file.

    Parameters
    ----------
    data : 3D numpy array
        The data to be written. Must have 3 dimensions, i.e. data.ndim == 3
    outputFilename : string
        The absolute or relative location of the particular file to be read
        in. outputFilename must end in one of the following extensions
        ['.tif', '.tiff', '.hdf5', '.h5', '.npy'].

    Notes
    -----
    - Data to be saved must be a 3D array.

    """

    assert data.ndim==3, "Can only write out 3D hdf5, tiff, and numpy files"
    filename = outputFilename.rstrip('/')
    basePath, fName = os.path.split(filename)
    name, ext = os.path.splitext(fName)
    if basePath and not os.path.exists(basePath):
        raise IOError, "Directory does not exist: %s" % (basePath)

    if ext.lower() in ['.npy']:
        try:
            np.save(filename, np.array(data,dtype=np.float32))
        except IOError:
            raise IOError, "Error writing npy data to: \"%s\"" % filename

    elif ext.lower() in ['.h5', '.hdf5']:
        from h5py import File
        try:
            h5File = File(filename, "w")
        except IOError:
            raise IOError, "Error creating writable hdf5 file at: \"%s\"" % filename

        shp = data.shape
        comp="gzip"
        compOpts=1
        dset = h5File.create_dataset("/raw", shp, np.float32, data, chunks=shp, compression=comp, compression_opts=compOpts)

    elif ext.lower() in ['.tif', '.tiff']:
        from libtiff import TIFF
        try:
            tiff = TIFF.open(filename, 'w')
            tiff.write_image(np.array(data,dtype=np.float32))
        except IOError:
            raise IOError, "Error writing tif file at: \"%s\"" % filename
        tiff.close()

    else:
        assert False, "Can only write out 3D hdf5, tiff, and numpy files"

def loadData(outputFilename, frames=None, ycrop=None, xcrop=None, transpose=False, datatype=np.float64):
    """
    Reads data from a tiff, hdf5, or npy file and returns a numpy array.

    Parameters
    ----------
    outputFilename : string
        The absolute or relative location of the particular hdf5 or
        tiff file to be read in. Filename must end in one of the following
        extensions ['tif', 'tiff', 'hdf5', 'h5', 'npy', 'lsm']. **If the file is
        hdf5, loadData assumes the data is in a dataset called \"/raw\"**.
    frames : array_like
        A tuple, list, or array containing the beginning and ending indices
        for the frames to be extracted. i.e. frames=[23,77] will return a
        numpy array containing the frames 23-76 within the file, 'filename'.
    xcrop : array_like
        A tuple, list, or array containing the beginning and ending indices
        for the columns to be extracted. This allows you to specify a subset
        or roi of the data to be returned. i.e. xcrop=[0,100] will return a
        numpy array containing first 100 columns of each frame within the file
        'filename'.
    ycrop : array_like
        A tuple, list, or array containing the beginning and ending indices
        for the rows to be extracted. This allows you to specify a subset
        or roi of the data to be returned. i.e. ycrop=[0,100] will return a
        numpy array containing first 100 rows of each frame within the file
        'filename'.
    transpose : boolean
        This specifies whether or not to transpose the last two dimensions
        of each frame. This might happen when reading and writing data between
        matlab and python, for example.
    dataset : numpy.dtype
        Specify the datatype of the returned numpy array. If dataset is of
        lower precision than the original data then truncation will take place.

    Returns
    -------
    filename_data : array
        The data read from within the file specified in filename

    Notes
    -----
    - If the file type is hdf5, loadData assumes the data to be read in is stored in
    a dataset called \"/raw\".
    - After the data has been read in, a .npy file is created and saved with a filename
    that specifies the parameters of the modified data. If in the future you wish to
    read in the same data, the .npy file will be read instead, saving time.

    Example
    -------

    >>> fname = "data/testHDF5.hdf5"
    >>> yroi = [175,250]
    >>> xroi = [100,150]
    >>> hdfData = loadData(fname, frames=[0,32], ycrop=yroi, xcrop=xroi)
    >>> hdfData.shape
    (32, 75, 50)

    >>> hdfData.dtype
    dtype('float32')

    >>> hdfData = loadData(fname, ycrop=yroi, xcrop=xroi, datatype=np.int16)
    >>> hdfData.shape
    (544, 75, 50)

    >>> hdfData.dtype
    dtype('int16')

    """

    print "-- Loading Data..."

    print "outputFilename = ", outputFilename
    filename = outputFilename.rstrip('/')
    print "filename = ", filename
    basePath, fName = os.path.split(filename)
    name, ext = os.path.splitext(fName)

    if basePath and not os.path.exists(basePath):
        raise IOError, "Directory does not exist: %s" % (basePath)
    elif not os.path.exists(filename):
        raise IOError, "File does not exist: %s" % (fName)

    npFilenameBase = os.path.join(basePath, name)
    if not frames is None:
        f0 = frames[0]; f1 = frames[1]
        npFilenameBase += "_frames" + str(f0) + "-" + str(f1-1)
    if not ycrop is None:
        y0 = ycrop[0]; y1 = ycrop[1]
        npFilenameBase += "_ycrop" + str(y0) + "-" + str(y1-1)
    if not xcrop is None:
        x0 = xcrop[0]; x1 = xcrop[1]
        npFilenameBase += "_xcrop" + str(x0) + "-" + str(x1-1)
    if transpose:
        npFilenameBase += "_T"
    npFilenameBase += "_" + str(np.dtype(datatype))

    # File name for the numpy file
    np_filename = npFilenameBase + '.npy'

    # Check if a numpy file already exists for this data
    if os.path.exists(np_filename):
        print "\t- Numpy file already exists. Loading %s..." % (np_filename)
        # If so, load it and be done
        try:
            volumeData = np.load(np_filename)
        except IOError:
            raise IOError, "Error reading filename: \"%s\"" % filename
        return volumeData

    elif ext.lower() in ['.npy']:  # If it is a numpy file
        try:
            volumeData = np.load(filename)
        except IOError:
            raise IOError, "Error reading filename: \"%s\"" % filename

    # Otherwise check if the data is in hdf5 format
    elif ext.lower() in ['.h5', '.hdf5']:
        from h5py import File
        print "\t- Reading from hdf5 file %s..." % (filename)

        try:
            h5File = File(filename, 'r')
        except IOError:
            raise IOError, "Error reading filename: \"%s\"" % (filename)

        volumeData = np.array(h5File["/raw"])

    # Then check to see if it is an lsm file
    elif ext.lower() in ['.lsm']:
        print "\n\nIN LSM SECTION\n\n"
        from libtiff import TIFFfile
        try:
            tiff = TIFFfile(filename)
        except IOError:
            raise IOError, "Error opening lsm file: \"%s\"" % (filename)
        samples, sample_names = tiff.get_samples()

        outList = []
        for sample in samples:
            outList.append(np.copy(sample)[...,np.newaxis])

        out = np.concatenate(outList,axis=-1)
        out = np.rollaxis(out,0,3)

        data = np.swapaxes(out[:,:,:,0],0,2)
        volumeData = np.swapaxes(data,1,2)

        tiff.close()

    # Then finally it must a tif file (hopefully)...
    elif ext.lower() in ['.tif', '.tiff']:
        print "\t- Reading from tiff file %s..." % (filename)

        from libtiff import TIFF
        try:
            tiff = TIFF.open(filename, 'r')
        except IOError:
            raise IOError, "Error opening tiff file: \"%s\"" % (filename)

        for count,image in enumerate(tiff.iter_images()):
            image_shape = image.shape

        volumeData = np.ndarray((count+1, image_shape[0], image_shape[1]), order='c', dtype=np.float64) # DO NOT HARDCODE

        for count,image in enumerate(tiff.iter_images()):
            volumeData[count,:,:] = image

    else:
        assert False, "The filename must have one of the following extensions [\"h5\", \"hdf5\", \"tif\", \"tiff\", \"npy\"]"

    if transpose:
        # For some reason the hdf5 data is transposed when written from matlab...
        volumeData = np.swapaxes(volumeData,1,2)
    dims = volumeData.shape
    if frames is None:
        f0 = 0; f1 = dims[0]
    if ycrop is None:
        y0 = 0; y1 = dims[1]
    if xcrop is None:
        x0 = 0; x1 = dims[2]

    print "made it to here!"
    finalData = np.array(volumeData[f0:f1,y0:y1,x0:x1], dtype=datatype)

    print "\t- Saving to numpy data file %s" % (np_filename)
    # Save it so we don't have to parse the tiff/hdf5 file every time
    np.save(npFilenameBase, finalData)

    return finalData
