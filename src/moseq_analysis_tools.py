import h5py
import pandas as pd


class MoseqProcessor:

    def __init__(self, file_path):
        """
        Initializes the Keypoint-Moseq processor class 

        Parameters:
        file_path (str): The path to the .h5 file containing the moseq data
        """

        self.file_path = file_path
        self.data = None 

    def load_data(self):
        """
        Loads MoSeq data from the h5 file and stores it in the 'data' attribute.
        
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        
        all_data = []

        with h5py.File(self.file_path, 'r') as hdf:
            for group_name in hdf.keys():
                group_data = hdf[group_name]

                # Extract data arrays
                centroid = group_data['centroid'][:]
                heading = group_data['heading'][:]
                latent_state = group_data['latent_state'][:]
                syllable = group_data['syllable'][:]

                # Create the main DataFrame
                df = pd.DataFrame({
                    'centroid_x': centroid[:, 0],
                    'centroid_y': centroid[:, 1],
                    'heading': heading,
                    'syllable': syllable
                })

                # Create and concatenate the latent state DataFrame
                latent_df = pd.DataFrame(latent_state, 
                                         columns=[f'latent_{i}' for i in 
                                                  range(latent_state.shape[1])])
                df = pd.concat([df, latent_df], axis=1)

                # Add cohort identifier
                df['cohort_id'] = group_name

                all_data.append(df)

        self.data = pd.concat(all_data, ignore_index=True)
        return self.data

