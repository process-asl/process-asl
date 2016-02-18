import os
import glob


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
        raise ValueError('Non exitant file with pattern {}'.format(pattern))

    if len(filenames) > 1:
        raise ValueError('Non unique file with pattern {}'.format(pattern))

    return filenames[0]


def load_heroes_dataset(
    subjects=None,
    subjects_parent_directory='/volatile/asl_data/heroes/raw',
    dataset_pattern={'anat': 't1mri/acquisition1/anat*.nii',
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

    dataset_pattern : dict, optional
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
    for (image_type, file_pattern) in dataset_pattern.iteritems():
        dataset[image_type] = []
        for subject_dir in subjects_directories:
            dataset[image_type].append(
                _single_glob(os.path.join(subject_dir, file_pattern)))
    return dataset
