import os
import glob
import warnings
import numpy as np
from sklearn.datasets.base import Bunch
from nilearn.datasets.utils import _fetch_files, _get_dataset_dir
from ._utils import _get_dataset_descr


def _single_glob(pattern):
    """Returns the file matching a given pattern. An error is raised if
    no file/multiple files match the pattern

    Parameters
    ----------
    pattern : str
        The pattern to match.

    Returns
    -------
    output : str or None
        The filename if existant.
    """
    filenames = glob.glob(pattern)
    if not filenames:
        raise ValueError('Non exitant file with pattern {0}'.format(pattern))

    if len(filenames) > 1:
        raise ValueError('Non unique file with pattern {0}'.format(pattern))

    return filenames[0]


def load_heroes_dataset(
    subjects=None,
    subjects_parent_directory='/volatile/asl_data/heroes/raw',
    paths_patterns={'anat': 't1mri/acquisition1/anat*.nii',
                    'basal ASL': 'fMRI/acquisition1/basal_rawASL*.nii',
                    'basal CBF': 'B1map/acquisition1/basal_relCBF*.nii'}
        ):
    """Loads the NeuroSpin HEROES dataset.

    Parameters
    ----------
    subjects : sequence of int or None, optional
        ids of subjects to load, default to loading all subjects.

    subjects_parent_directory : str, optional
        Path to the dataset folder containing all subjects folders.

    paths_patterns : dict, optional
        Input dictionary. Keys are the names of the images to load, values
        are strings specifying the unique relative pattern specifying the
        path to these images within each subject directory.

    Returns
    -------
    dataset : dict
        The absolute paths to the images for all subjects. Keys are the same
        as the files_patterns keys, values are lists of strings.
    """
    # Absolute paths of subjects folders
    subjects_directories = [os.path.join(subjects_parent_directory, name)
                            for name in
                            sorted(os.listdir(subjects_parent_directory))
                            if os.path.isdir(os.path.join(
                                subjects_parent_directory, name))]
    max_subjects = len(subjects_directories)
    if subjects is None:
        subjects = range(max_subjects)
    else:
        if max(subjects) > max_subjects:
            raise ValueError('Got {0} subjects, you provided ids {1}'
                             ''.format(max_subjects, str(subjects)))

    subjects_directories = [subjects_directories[subject_id] for subject_id in
                            subjects]

    # Build the path list for each image type
    dataset = {}
    for (image_type, file_pattern) in paths_patterns.iteritems():
        dataset[image_type] = []
        for subject_dir in subjects_directories:
            dataset[image_type].append(
                _single_glob(os.path.join(subject_dir, file_pattern)))
    return dataset


def fetch_kirby(subjects=range(2), sessions=[1], data_dir=None, url=None,
                resume=True, verbose=1):
    """Download and load the KIRBY multi-modal dataset.

    Parameters
    ----------
    subjects : sequence of int or None, optional
        ids of subjects to load, default to loading 2 subjects.

    sessions: iterable of int, optional
        The sessions to load. Load only the first session by default.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data). Default: None

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'anat': Paths to structural MPRAGE images
         - 'asl': Paths to ASL images
         - 'm0': Paths to ASL M0 images

    Notes
    ------

    This dataset is composed of 2 sessions of 21 participants (11 males) at 3T.
    Imaging modalities include MPRAGE, FLAIR,
    DTI, resting state fMRI, B0 and B1 field maps, ASL, VASO, quantitative T1
    mapping, quantitative T2 mapping, and magnetization transfer imaging.
    For each session, we only download MPRAGE and ASL data.

    More details about this dataset can be found here :
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3020263
    http://mri.kennedykrieger.org/databases.html

    Paper to cite
    -------------
        `Multi-Parametric Neuroimaging Reproducibility: A 3T Resource Study
        <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3020263>`_
        Bennett. A. Landman, Alan J. Huang, Aliya Gifford,Deepti S. Vikram,
        Issel Anne L. Lim, Jonathan A.D. Farrell, John A. Bogovic, Jun Hua,
        Min Chen,
        Samson Jarso, Seth A. Smith, Suresh Joel, Susumu Mori, James J. Pekar,
        Peter B. Barker, Jerry L. Prince, and Peter C.M. van Zijl.
        NeuroImage. (2010)
        NIHMS/PMC:252138 doi:10.1016/j.neuroimage.2010.11.047

    Licence
    -------
    `BIRN Data License
    <http://www.nbirn.net/bdr/Data_Use_Agreement_09_19_07-1.pdf>`_
    """

    if url is None:
        url = 'https://www.nitrc.org/frs/downloadlink.php/'

    # Preliminary checks and declarations
    dataset_name = 'kirby'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    subject_ids = np.array([
        '849', '934', '679', '906', '913', '142', '127', '742', '422', '815',
        '906', '239', '916', '959', '814', '505', '959', '492', '239', '142',
        '815', '679', '800', '916', '849', '814', '800', '656', '742', '113',
        '913', '502', '113', '127', '505', '502', '934', '492', '346', '656',
        '346', '422'])
    nitrc_ids = np.arange(2201, 2243)
    ids = np.arange(1, 43)

    # Group indices by session
    _, indices1 = np.unique(subject_ids, return_index=True)
    subject_ids1 = subject_ids[sorted(indices1)]
    nitrc_ids1 = nitrc_ids[sorted(indices1)]
    ids1 = ids[sorted(indices1)]

    tuple_indices = [np.where(subject_ids == s)[0] for s in subject_ids1]
    indices2 = [idx1 if idx1 not in indices1 else idx2
                for (idx1, idx2) in tuple_indices]
    subject_ids2 = subject_ids[indices2]
    nitrc_ids2 = nitrc_ids[indices2]
    ids2 = ids[indices2]

    # Check arguments
    max_subjects = len(subject_ids)
    if max(subjects) > max_subjects:
        warnings.warn('Warning: there are only {0} subjects'.format(
            max_subjects))
        subjects = range(max_subjects)
    unique_subjects, indices = np.unique(subjects, return_index=True)
    if len(unique_subjects) < len(subjects):
        warnings.warn('Warning: Duplicate subjects, removing them.')
        subjects = unique_subjects[np.argsort(indices)]

    n_subjects = len(subjects)

    archives = [
        [url + '{0}/KKI2009-{1:02}.tar.bz2'.format(nitrc_id, id) for
         (nitrc_id, id) in zip(nitrc_ids1, ids1)],
        [url + '{0}/KKI2009-{1:02}.tar.bz2'.format(nitrc_id, id) for
         (nitrc_id, id) in zip(nitrc_ids2, ids2)]
                ]
    anat1 = [os.path.join('session1', subject,
                          'KKI2009-{0:02}-MPRAGE.nii'.format(i))
             for subject, i in zip(subject_ids1, ids1)]
    anat2 = [os.path.join('session2', subject,
                          'KKI2009-{0:02}-MPRAGE.nii'.format(i))
             for subject, i in zip(subject_ids2, ids2)]
    asl1 = [os.path.join('session1', subject,
                         'KKI2009-{0:02}-ASL.nii'.format(i))
            for subject, i in zip(subject_ids1, ids1)]
    asl2 = [os.path.join('session2', subject,
                         'KKI2009-{0:02}-ASL.nii'.format(i))
            for subject, i in zip(subject_ids2, ids2)]
    m01 = [os.path.join('session1', subject,
                        'KKI2009-{0:02}-ASLM0.nii'.format(i))
           for subject, i in zip(subject_ids1, ids1)]
    m02 = [os.path.join('session2', subject,
                        'KKI2009-{0:02}-ASLM0.nii'.format(i))
           for subject, i in zip(subject_ids2, ids2)]

    target = [
        [os.path.join('session1', subject, 'KKI2009-{0:02}.tar.bz2'.format(id))
         for (subject, id) in zip(subject_ids1, ids1)],
        [os.path.join('session2', subject, 'KKI2009-{0:02}.tar.bz2'.format(id))
         for (subject, id) in zip(subject_ids2, ids2)]
                ]
    anat = [anat1, anat2]
    asl = [asl1, asl2]
    m0 = [m01, m02]

    source_anat = []
    source_asl = []
    source_m0 = []
    source_archives = []
    session = []
    target_archives = []
    for i in sessions:
        if not (i in [1, 2]):
            raise ValueError('KIRBY dataset session id must be in [1, 2]')
        source_anat += [anat[i - 1][subject] for subject in subjects]
        source_asl += [asl[i - 1][subject] for subject in subjects]
        source_m0 += [m0[i - 1][subject] for subject in subjects]
        source_archives += [archives[i - 1][subject] for subject in subjects]
        target_archives += [target[i - 1][subject] for subject in subjects]

        session += [i] * n_subjects

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # Call fetch_files once per subject.
    asl = []
    m0 = []
    anat = []
    for anat_u, asl_u, m0_u, archive, target in zip(source_anat, source_asl,
                                                    source_m0, source_archives,
                                                    target_archives):
        n, a, m = _fetch_files(
            data_dir,
            [(anat_u, archive, {'uncompress': True, 'move': target}),
             (asl_u, archive, {'uncompress': True, 'move': target}),
             (m0_u, archive, {'uncompress': True, 'move': target})],
            verbose=verbose)

        anat.append(n)
        asl.append(a)
        m0.append(m)

    return Bunch(anat=anat, asl=asl, m0=m0, description=fdescr)
