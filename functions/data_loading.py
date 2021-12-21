import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split



class DataLoad():

    def __init__(self):
        self.pca_variables = [
                'Present_Tmax',
                'Present_Tmin',
                'LDAPS_RHmin',
                'LDAPS_RHmax',
                'LDAPS_Tmax_lapse',
                'LDAPS_Tmin_lapse',
                'LDAPS_WS',
                'LDAPS_LH',
                'LDAPS_CC1',
                'LDAPS_CC2',
                'LDAPS_CC3',
                'LDAPS_CC4',
                'LDAPS_PPT1',
                'LDAPS_PPT2',
                'LDAPS_PPT3',
                'LDAPS_PPT4',
                'Solar radiation',
                  ]
        
        self.input_variables = self.pca_variables.copy()
        [self.input_variables.append(variable) for variable in ['pca 0', 'pca 1', 'pca 2']]

        self.output_variables = [
                'Next_Tmax',
                'Next_Tmin'
        ]

        self.df = pd.read_csv('./data/Bias_correction_ucl.csv')
        
        self.df_station = dict()
        self.df_train = dict()
        self.df_test = dict()
        self.y_train = dict()
        self.y_test = dict()


        self.clean()
        self.add_pca()
        self.split_train_validate()


    def clean(self):
        self.df = self.df[self.df['Present_Tmax'].notna() & self.df['Present_Tmin'].notna() & 
        self.df['Next_Tmax'].notna() & self.df['Next_Tmin'].notna() & 
        self.df['station'].notna() & self.df['Date'].notna() & 
        self.df['LDAPS_RHmin'].notna()]

        for i in range(1, 26):
            self.df_station[i] = self.df[self.df['station'] == i].copy()


    def add_pca(self, comps=3):
        pca = PCA(n_components=comps)

        for i in range(1, 26):
            projection = pca.fit_transform(self.df_station[i][self.pca_variables])
            for k in range(0, projection[0].size):
                self.df_station[i][f'pca {k}'] = projection[:,k]


    def split_train_validate(self, val_prop=0.2):
        # Need to change the split method if approaching the problem as a timeseries. Using a random splitter now
        for i in range(1, 26):
            df_train_interm = self.df_station[i].sample(frac=0.8, random_state=47)
            df_test_interm = self.df_station[i].drop(df_train_interm.index)

            self.df_train[i] = df_train_interm[self.input_variables].copy()
            self.y_train[i] = df_train_interm[self.output_variables].copy()

            self.df_test[i] = df_test_interm[self.input_variables].copy()
            self.y_test[i] = df_test_interm[self.output_variables].copy()
