from ranking_pipeline.synthetic import generate_synthetic_interactions
from ranking_pipeline.split import split_interactions
from ranking_pipeline.train import train_pointwise_regressor, predict_relevance


def test_train_predict_shape():
    df = generate_synthetic_interactions(seed=99)
    train_df, eval_df, _ = split_interactions(df, seed=99)
    art = train_pointwise_regressor(train_df, seed=99)
    preds = predict_relevance(art, eval_df)
    assert len(preds) == len(eval_df)
