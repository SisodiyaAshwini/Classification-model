using Microsoft.ML.Data;

namespace MachineLearning.Common.Interfaces
{
    public interface ITrainerBase
    {
        string Name { get; }
        void Fit(string trainingFileName);
        MulticlassClassificationMetrics Evaluate();
        void Save();
    }
}
