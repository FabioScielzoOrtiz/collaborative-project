import polars as pl
from PyMachineLearning.preprocessing import encoder

def processing(df):
    columns_to_exclude = ['', 'id','sq_mt_allotment','floor', 'neighborhood', 'district'] 
    df = df.select(pl.exclude(columns_to_exclude))
    binary_cols = ['is_renewal_needed', 'has_lift', 'is_exterior', 'has_parking']
    multi_cols = ['energy_certificate', 'house_type']
    quant_cols = [x for x in df.columns if x not in binary_cols + multi_cols]
    encoding = encoder(method='ordinal')
    encoded_arr = encoding.fit_transform(df[binary_cols + multi_cols])
    cat_df = pl.DataFrame(encoded_arr)
    cat_df.columns =  binary_cols + multi_cols
    cat_df = cat_df.with_columns([pl.col(col).cast(pl.Int64) for col in cat_df.columns])
    quant_df = df[quant_cols]
    df = pl.concat([quant_df, cat_df], how='horizontal')
    response = 'buy_price'
    quant_predictors = [x for x in quant_cols if x != response]
    binary_predictors = [x for x in binary_cols if x != response]
    multi_predictors = [x for x in multi_cols if x != response]
    cat_predictors = binary_predictors + multi_predictors
    p1, p2, p3 = len(quant_predictors), len(binary_predictors), len(multi_predictors)
    return df, p1, p2, p3, response, quant_predictors, cat_predictors

madrid_houses_df = pl.read_csv('data/madrid_houses.csv')
madrid_houses_df, p1, p2, p3, response, quant_predictors, cat_predictors = processing(madrid_houses_df)
madrid_houses_df = madrid_houses_df.filter(pl.col('buy_price') > 3000000)
madrid_houses_df.write_csv('./data/madrid_houses_processed.csv', separator=",")
print('Execution finished correctly.')