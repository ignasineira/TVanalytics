import pandas as pd

    def pipeline(merge_canal = False):
    df = pd.read_csv('input/random_data/data_ejemplo_input_pronostico_v3.csv', sep=';')
    df['DIA_HORA'] = pd.to_datetime(df.DIA_HORA)
    df.sort_values(by=['DIA_HORA'], inplace=True, ascending=False)
    df.drop(columns=["LLAVE_SS", 'DIA', 'DIA_HORA'], inplace=True)
    to_covert_cat = [ 'CANAL_GENERO', 'CANAL_TARGET_DEMO', 'FERIADO', 'DIA_IDX', 'PROGRAMA_ID1',
                     'PROGRAMA_NOMBRE',
                     'EPISODIO_NOMBRE', 'EPISODIO_ORIGINAL_ADQUERIDO_CODIGO',
                     'TIPO1_PROGRAMA', 'TIPO2_PROGRAMA', 'GENERO']
    for col in to_covert_cat:
        df[col] = df[col].astype('category')

    # to_convert_date=['DIA_HORA'] #pasarla la hora del dia a minutos

    to_covert_num = ['ANIO', 'MES', 'SEMANA_EN_MES', 'DIA_SEMANA', 'HR_ID',
                     'PROGRAMA_DURACION', 'IF_REPETIDO', 'ANIO_ESTRENO', 'AP', 'UNIV_CANAL_PRMD', 'SHARE', 'RATING',
                     'AP_PROPF', 'AP_PROP4', 'AP_PROP7', 'AP_PROP10', 'AP_PROP13',
                     'AP_PROP16', 'AP_PROP19', 'AP_PROP23', 'AP_PROP27', 'AP_PROP32',
                     'AP_PROP37', 'AP_PROP42', 'AP_PROP47', 'AP_PROP52', 'AP_PRMDLAST4',
                     'AP_PRMDLAST8', 'AP_PRMDLAST12', 'AP_PRMDALLPAST', 'AP_STDDEVLAST12',
                     'SHARE_PRMDLAST4', 'SHARE_PRMDLAST8', 'SHARE_PRMDLAST12',
                     'SHARE_PRMDALLPAST', 'SHARE_STDDEVLAST12', 'RATING_PRMDLAST4',
                     'RATING_PRMDLAST8', 'RATING_PRMDLAST12', 'RATING_PRMDALLPAST',
                     'RATING_STDDEVLAST12', 'AP_PROPF_PRMDLAST4', 'AP_PROPF_PRMDLAST8',
                     'AP_PROPF_PRMDLAST12', 'AP_PROPF_PRMDALLPAST', 'AP_PROPF_STDDEVLAST12',
                     'AP_PROP4_PRMDLAST4', 'AP_PROP7_PRMDLAST4', 'AP_PROP10_PRMDLAST4',
                     'AP_PROP13_PRMDLAST4', 'AP_PROP16_PRMDLAST4', 'AP_PROP19_PRMDLAST4',
                     'AP_PROP23_PRMDLAST4', 'AP_PROP27_PRMDLAST4', 'AP_PROP32_PRMDLAST4',
                     'AP_PROP37_PRMDLAST4', 'AP_PROP42_PRMDLAST4', 'AP_PROP47_PRMDLAST4',
                     'AP_PROP52_PRMDLAST4']
    for col in to_covert_num:
        df[col] = df[col].astype('float64')
    df = pd.get_dummies(df, columns=to_covert_cat)

    lista_canales = df.CANAL.unique()
    #merge_canal = False
    if merge_canal == True:
        # aux=df.groupby(['CANAL', 'ANIO', 'MES', 'SEMANA_EN_MES', 'DIA_SEMANA', 'HR_ID','PROGRAMA_DURACION', 'IF_REPETIDO', 'ANIO_ESTRENO']).agg("mean")
        # aux.reset_index(inplace=True)
        for canal in lista_canales:
            print(df[df.CANAL == canal].shape)

    elif merge_canal== False:
        #df = df[df.CANAL == lista_canales[0]]
        df = df[df.CANAL == 'AAA']
        df.drop(columns=['CANAL'], inplace=True)

    lista_columns = ['AP', 'UNIV_CANAL_PRMD', 'SHARE', 'RATING',
                     'AP_PROPF', 'AP_PROP4', 'AP_PROP7', 'AP_PROP10', 'AP_PROP13',
                     'AP_PROP16', 'AP_PROP19', 'AP_PROP23', 'AP_PROP27', 'AP_PROP32',
                     'AP_PROP37', 'AP_PROP42', 'AP_PROP47', 'AP_PROP52']

    df_X = df[df.columns.difference(lista_columns)].copy()
    df_y = df[lista_columns].copy()

    return df, df_X, df_y