using Microsoft.ML.Data;

namespace MachineLearning.DataModels
{
    public class ElementPrediction
    {
        [ColumnName("PredictedLabel")]
        public string? Element;
    }
}
