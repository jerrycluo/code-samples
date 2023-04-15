import os
import pyspark.sql.functions as F
import pyspark.sql.types as T
from utilities import SEED
# availiable on AWS EMR

# ---------------- choose input format, dataframe or rdd ----------------------
INPUT_FORMAT = 'dataframe' 
# -----------------------------------------------------------------------------
if INPUT_FORMAT == 'dataframe':
    import pyspark.ml as M
    import pyspark.sql.functions as F
    import pyspark.sql.types as T
    from pyspark.ml.regression import DecisionTreeRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
if INPUT_FORMAT == 'koalas':
    import databricks.koalas as ks
elif INPUT_FORMAT == 'rdd':
    import pyspark.mllib as M
    from pyspark.mllib.feature import Word2Vec
    from pyspark.mllib.linalg import Vectors
    from pyspark.mllib.linalg.distributed import RowMatrix
    from pyspark.mllib.tree import DecisionTree
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.linalg import DenseVector
    from pyspark.mllib.evaluation import RegressionMetrics



def task_0('user_reviews_Release.csv'):
    from dask.distributed import Client
    client = Client()
    client = client.restart()
    
    import dask.dataframe as dd
    df = dd.read_csv('user_reviews_Release.csv')
    
    df = df.drop(columns = ['reviewerName','reviewText','unixReviewTime','summary','asin'])

    df['total'] = df['helpful'].apply(lambda s: int(s.split(',')[1][1:-1]), meta = 'int64')
    df['helpful'] = df['helpful'].apply(lambda s: int(s.split(',')[0][1:]), meta = ('total','int64'))
    df['reviewTime'] = df['reviewTime'].apply(lambda x: int(x[-4:]), meta = ('reviewTime', 'int64'))

    group = df.groupby('reviewerID')
    num_products_rated = group['helpful'].count(split_out=8)
    avg_ratings = group['overall'].mean(split_out=8)
    total_votes = group['total'].sum(split_out=8)
    helpful_votes = group['helpful'].sum(split_out=8)
    reviewing_since = group['reviewTime'].min(split_out=8)

    new = dd.concat([num_products_rated,avg_ratings],axis = 1)
    new = new.rename(columns = {'helpful': 'number_products_rated', 'overall': 'avg_ratings'})
    new = dd.concat([new,reviewing_since,helpful_votes, total_votes],axis = 1)
    new = new.rename(columns = {'reviewTime':'reviewing_since', 'helpful':'helpful_votes', 'total': 'total_votes'})

    submit = new.describe().compute().round(2)    
    with open('results_PA0.json', 'w') as outfile: json.dump(json.loads(submit.to_json()), outfile)



def task_1(data_io, review_data, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    overall_column = 'overall'
    # Outputs:
    mean_rating_column = 'meanRating'
    count_rating_column = 'countRating'
    # -------------------------------------------------------------------------

    
    
    a = product_data.join(review_data,product_data.asin==review_data.asin,'left').drop(review_data.asin)
    
    count = a[['asin','overall']].na.drop().groupby('asin').count()
    mean = a.groupBy('asin').mean('overall')

    b = product_data
    b = b.join(count,b.asin==count.asin,'left').drop(count.asin)
    b = b.join(mean,b.asin==mean.asin,'left').drop(mean.asin)

    b = b.withColumn('meanRating',F.col('avg(overall)'))
    b = b.withColumn('countRating',F.col('count'))
    


    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    # Modify the values of the following dictionary accordingly.
    res = {
        'count_total': None,
        'mean_meanRating': None,
        'variance_meanRating': None,
        'numNulls_meanRating': None,
        'mean_countRating': None,
        'variance_countRating': None,
        'numNulls_countRating': None
    }
    # Modify res:
    res['count_total'] = b.count()
    res['mean_meanRating'] = b.select(F.avg(F.col('meanRating'))).head(1)[0][0]
    res['variance_meanRating'] = b.select(F.variance(F.col('meanRating'))).head(1)[0][0]
    res['numNulls_meanRating'] = b.select(F.count(F.when(F.col('meanRating').isNull(),'meanRating'))).head(1)[0][0]
    res['mean_countRating'] = b.select(F.avg(F.col('countRating'))).head(1)[0][0]
    res['variance_countRating'] = b.select(F.variance(F.col('countRating'))).head(1)[0][0]
    res['numNulls_countRating'] = b.select(F.count(F.when(F.col('countRating').isNull(),'countRating'))).head(1)[0][0]



    # -------------------------------------------------------------------------

    data_io.save(res, 'task_1')
    return res
    # -------------------------------------------------------------------------


def task_2(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    salesRank_column = 'salesRank'
    categories_column = 'categories'
    asin_column = 'asin'
    # Outputs:
    category_column = 'category'
    bestSalesCategory_column = 'bestSalesCategory'
    bestSalesRank_column = 'bestSalesRank'
    # -------------------------------------------------------------------------


    a = product_data

    def y(x):
        if x is None:
            return None
        if len(x[0]) == 0:
            return None
        elif len(x[0][0]) == 0:
            return None
        else:
            return x[0][0]

    genCat = F.udf(y, T.StringType())

    a = a.withColumn(category_column,genCat(F.col(categories_column)))

    def y1(x):
        if x is None:
            return None
        elif len(list(x.keys())) == 0:
            return None
        else:
            return list(x.keys())[0]
    
    def y2(x):
        if x is None:
            return None
        elif len(list(x.values())) == 0:
            return None
        else:
            return list(x.values())[0]


    rank = F.udf(y2,T.IntegerType())
    cat = F.udf(y1, T.StringType())

    a = a.withColumn(bestSalesRank_column,rank(F.col(salesRank_column)))
    a = a.withColumn(bestSalesCategory_column,cat(F.col(salesRank_column)))
    

    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_bestSalesRank': None,
        'variance_bestSalesRank': None,
        'numNulls_category': None,
        'countDistinct_category': None,
        'numNulls_bestSalesCategory': None,
        'countDistinct_bestSalesCategory': None
    }
    # Modify res:


    res['count_total'] = a.count()
    res['mean_bestSalesRank'] = a.select(F.avg(F.col('bestSalesRank'))).head(1)[0][0]
    res['variance_bestSalesRank'] = a.select(F.variance(F.col('bestSalesRank'))).head(1)[0][0]
    res['numNulls_category'] = a.select(F.count(F.when(F.col('category').isNull(),'category'))).head(1)[0][0]
    res['countDistinct_category'] = a[['category']].distinct().count()-1
    res['numNulls_bestSalesCategory'] = a.select(F.count(F.when(F.col('bestSalesCategory').isNull(),'category'))).head(1)[0][0]
    res['countDistinct_bestSalesCategory'] = a[['bestSalesCategory']].distinct().count()-1

    # -------------------------------------------------------------------------


    data_io.save(res, 'task_2')
    return res
    # -------------------------------------------------------------------------



def task_3(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    asin_column = 'asin'
    price_column = 'price'
    attribute = 'also_viewed'
    related_column = 'related'
    # Outputs:
    meanPriceAlsoViewed_column = 'meanPriceAlsoViewed'
    countAlsoViewed_column = 'countAlsoViewed'
    # -------------------------------------------------------------------------

    
    a = product_data

    def y1(x):
        if x is None:
            return None
        if 'also_viewed' not in x.keys():
            return None
        return x['also_viewed']

    alsoViewed = F.udf(y1, T.ArrayType(T.StringType()))

    a = a.withColumn('also_viewed',alsoViewed(F.col('related')))

    b = a.select(asin_column, F.explode_outer(attribute).alias('viewed'))

    b = b.withColumnRenamed(asin_column, 'original')

    c = product_data.join(b, product_data.asin ==b.viewed,'right')

    g = c.groupby('original').mean('price')

    g = g.withColumn('meanPriceAlsoViewed', F.col('avg(price)'))

    f = product_data.join(g, g.original==product_data.asin)

    def y2(x):
        if x is None:
            return None
        if 'also_viewed' not in x.keys():
            return None
        return len(x['also_viewed'])

    countAlsoViewed = F.udf(y2, T.IntegerType())

    f = f.withColumn('countAlsoViewed',countAlsoViewed(F.col('related'))) 




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanPriceAlsoViewed': None,
        'variance_meanPriceAlsoViewed': None,
        'numNulls_meanPriceAlsoViewed': None,
        'mean_countAlsoViewed': None,
        'variance_countAlsoViewed': None,
        'numNulls_countAlsoViewed': None
    }
    # Modify res:
    
    res['count_total'] = f.count()
    res['mean_meanPriceAlsoViewed'] = f.select(F.avg(F.col('meanPriceAlsoViewed'))).head(1)[0][0]
    res['variance_meanPriceAlsoViewed'] = f.select(F.variance(F.col('meanPriceAlsoViewed'))).head(1)[0][0]
    res['numNulls_meanPriceAlsoViewed'] = f.select(F.count(F.when(F.col('meanPriceAlsoViewed').isNull(),\
                                                                  'meanPriceAlsoViewed'))).head(1)[0][0]
    res['mean_countAlsoViewed'] = f.select(F.avg(F.col('countAlsoViewed'))).head(1)[0][0]
    res['variance_countAlsoViewed'] = f.select(F.variance(F.col('countAlsoViewed'))).head(1)[0][0]
    res['numNulls_countAlsoViewed'] = f.select(F.count(F.when(F.col('countAlsoViewed').isNull(),\
                                                                  'countAlsoViewed'))).head(1)[0][0]


    # -------------------------------------------------------------------------

    data_io.save(res, 'task_3')
    return res
    # -------------------------------------------------------------------------



def task_4(data_io, product_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    price_column = 'price'
    title_column = 'title'
    # Outputs:
    meanImputedPrice_column = 'meanImputedPrice'
    medianImputedPrice_column = 'medianImputedPrice'
    unknownImputedTitle_column = 'unknownImputedTitle'
    # -------------------------------------------------------------------------

    
    from pyspark.ml.feature import Imputer

    a = product_data

    a = a.withColumn('unknownImputedTitle',F.col('title'))
    a = a.na.fill('unknown',['unknownImputedTitle'])

    imputer = Imputer(
        inputCols=['price'], 
        outputCols=["meanImputed{}".format(c) for c in ['Price']]
        ).setStrategy("mean")

    a = imputer.fit(a).transform(a) 

    imputer1 = Imputer(
        inputCols=['price'], 
        outputCols=["medianImputed{}".format(c) for c in ['Price']]
        ).setStrategy("median")

    a = imputer1.fit(a).transform(a) 



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'mean_meanImputedPrice': None,
        'variance_meanImputedPrice': None,
        'numNulls_meanImputedPrice': None,
        'mean_medianImputedPrice': None,
        'variance_medianImputedPrice': None,
        'numNulls_medianImputedPrice': None,
        'numUnknowns_unknownImputedTitle': None
    }
    # Modify res:

    res['count_total'] = a.count()
    res['mean_meanImputedPrice'] = a.select(F.avg(F.col('meanImputedPrice'))).head(1)[0][0]
    res['numNulls_meanImputedPrice'] = a.select(F.count(F.when(F.col('meanImputedPrice').isNull(),'meanImputedPrice'))).head(1)[0][0]
    res['variance_meanImputedPrice'] = a.select(F.variance(F.col('meanImputedPrice'))).head(1)[0][0]
    res['mean_medianImputedPrice'] = a.select(F.avg(F.col('medianImputedPrice'))).head(1)[0][0]
    res['variance_medianImputedPrice'] = a.select(F.variance(F.col('medianImputedPrice'))).head(1)[0][0]
    res['numNulls_medianImputedPrice'] = a.select(F.count(F.when(F.col('medianImputedPrice').isNull(),'meanImputedPrice'))).head(1)[0][0]
    res['numUnknowns_unknownImputedTitle'] = a.select(F.count(F.when(F.col('unknownImputedTitle') == 'unknown','unknownImputedTitle'))).head(1)[0][0]


    # -------------------------------------------------------------------------

    # ----------------------------- Do not change -----------------------------
    data_io.save(res, 'task_4')
    return res
    # -------------------------------------------------------------------------



def task_5(data_io, product_processed_data, word_0, word_1, word_2):
    # -----------------------------Column names--------------------------------
    # Inputs:
    title_column = 'title'
    # Outputs:
    titleArray_column = 'titleArray'
    titleVector_column = 'titleVector'
    # -------------------------------------------------------------------------


    a = product_processed_data
    y = lambda x: None if x is None else x.lower().split(' ')
    split = F.udf(y,T.ArrayType(T.StringType()))
    a = a.withColumn('titleArray',split(F.col('title')))
    
    v = M.feature.Word2Vec(minCount=100, vectorSize=16, seed=102, numPartitions=4,
                      inputCol = titleArray_column)
    model = v.fit(a)
        



    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'size_vocabulary': None,
        'word_0_synonyms': [(None, None), ],
        'word_1_synonyms': [(None, None), ],
        'word_2_synonyms': [(None, None), ]
    }
    # Modify res:
    res['count_total'] = a.count()
    res['size_vocabulary'] = model.getVectors().count()
    for name, word in zip(
        ['word_0_synonyms', 'word_1_synonyms', 'word_2_synonyms'],
        [word_0, word_1, word_2]
    ):
    
        res[name] = model.findSynonymsArray(word, 10)
    # -------------------------------------------------------------------------

    data_io.save(res, 'task_5')
    return res
    # -------------------------------------------------------------------------



def task_6(data_io, product_processed_data):
    # -----------------------------Column names--------------------------------
    # Inputs:
    category_column = 'category'
    # Outputs:
    categoryIndex_column = 'categoryIndex'
    categoryOneHot_column = 'categoryOneHot'
    categoryPCA_column = 'categoryPCA'
    # -------------------------------------------------------------------------    


    from pyspark.ml.stat import Summarizer

    a = product_processed_data
    indexer = M.feature.StringIndexer(inputCol='category',outputCol='indexer')
    a = indexer.fit(a).transform(a)
    encoder = M.feature.OneHotEncoder(inputCol='indexer',outputCol='categoryOneHot',dropLast=False)
    a = encoder.fit(a).transform(a)

    pca = M.feature.PCA(k=15,inputCol = 'categoryOneHot', outputCol = 'categoryPCA')
    a = pca.fit(a).transform(a)




    # -------------------------------------------------------------------------

    # ---------------------- Put results in res dict --------------------------
    res = {
        'count_total': None,
        'meanVector_categoryOneHot': [None, ],
        'meanVector_categoryPCA': [None, ]
    }
    # Modify res:

    res['count_total'] = a.count()
    res['meanVector_categoryOneHot'] = a[['categoryOneHot']].select(s.summary(a.categoryOneHot)).head(1)[0][0][0]
    res['meanVector_categoryPCA'] = a[['categoryPCA']].select(s.summary(a.categoryPCA)).head(1)[0][0][0]


    # -------------------------------------------------------------------------

    data_io.save(res, 'task_6')
    return res
    # -------------------------------------------------------------------------

    
def task_7(data_io, train_data, test_data):
    
    
    
    train_data = train_data.withColumn('label',F.col('overall')).drop('overall')
    from pyspark.ml.regression import DecisionTreeRegressor
    dt = DecisionTreeRegressor(maxDepth=5)
    model = dt.fit(train_data)
    p = model.transform(test_data)
    p = p.withColumn('label',F.col('overall'))
    
    from pyspark.ml.evaluation import RegressionEvaluator as re
    evaluator = re()
    

    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None
    }
    # Modify res:
    res['test_rmse'] = evaluator.evaluate(p,{evaluator.metricName: "rmse"})

    # -------------------------------------------------------------------------

    data_io.save(res, 'task_7')
    return res
    # -------------------------------------------------------------------------
    
def task_8(data_io, train_data, test_data):
    
    
    from pyspark.ml.evaluation import RegressionEvaluator as re
    from pyspark.ml.regression import DecisionTreeRegressor

    train_data,Validation_data = train_data.randomSplit([0.75, 0.25], 24)
    
    d = {}
    
    a,Validation_data = train_data.randomSplit([0.75, 0.25], 24)
    a = a.withColumn('label',F.col('overall')).drop('overall')
    dt = DecisionTreeRegressor(maxDepth=5)
    model5 = dt.fit(a)
    p5 = model5.transform(Validation_data)
    p5 = p5.withColumn('label',F.col('overall'))

    evaluator = re()
    d[5] = evaluator.evaluate(p5,{evaluator.metricName: "rmse"})
    
    a,Validation_data = train_data.randomSplit([0.75, 0.25], 24)
    a = a.withColumn('label',F.col('overall')).drop('overall')
    dt = DecisionTreeRegressor(maxDepth=7)
    model7 = dt.fit(a)
    p7 = model7.transform(Validation_data)
    p7 = p7.withColumn('label',F.col('overall'))

    evaluator = re()
    d[7] = evaluator.evaluate(p7,{evaluator.metricName: "rmse"})

    a,Validation_data = train_data.randomSplit([0.75, 0.25], 24)
    a = a.withColumn('label',F.col('overall')).drop('overall')
    dt = DecisionTreeRegressor(maxDepth=9)
    model9 = dt.fit(a)
    p9 = model9.transform(Validation_data)
    p9 = p9.withColumn('label',F.col('overall'))

    evaluator = re()
    d[9] = evaluator.evaluate(p,{evaluator.metricName: "rmse"})
    
    a,Validation_data = train_data.randomSplit([0.75, 0.25], 24)
    a = a.withColumn('label',F.col('overall')).drop('overall')
    dt = DecisionTreeRegressor(maxDepth=12)
    model12 = dt.fit(a)
    p12 = model12.transform(Validation_data)
    p12 = p12.withColumn('label',F.col('overall'))

    evaluator = re()
    d[12] = evaluator.evaluate(p,{evaluator.metricName: "rmse"})

    
    depth = min(d, key = d.get)
    if depth == 12:
        pb = model12.transform(test_data)
        pb = pb.withColumn('label',F.col('overall'))
    elif depth == 9:
        pb = model9.transform(test_data)
        pb = pb.withColumn('label',F.col('overall'))
    elif depth == 7:
        pb = model7.transform(test_data)
        pb = pb.withColumn('label',F.col('overall'))
    else:
        pb = model5.transform(test_data)
        pb = pb.withColumn('label',F.col('overall'))

    from pyspark.ml.evaluation import RegressionEvaluator as re
    evaluator = re()
    bestRMSE = evaluator.evaluate(pb,{evaluator.metricName: "rmse"})

    
    # -------------------------------------------------------------------------
    
    
    # ---------------------- Put results in res dict --------------------------
    res = {
        'test_rmse': None,
        'valid_rmse_depth_5': None,
        'valid_rmse_depth_7': None,
        'valid_rmse_depth_9': None,
        'valid_rmse_depth_12': None,
    }
    # Modify res:
    
    res['test_rmse'] = bestRMSE
    res['valid_rmse_depth_5'] = d[5]
    res['valid_rmse_depth_7'] = d[7]
    res['valid_rmse_depth_9'] = d[9]
    res['valid_rmse_depth_12'] = d[12]
    
    # -------------------------------------------------------------------------

    data_io.save(res, 'task_8')
    return res
    # -------------------------------------------------------------------------

