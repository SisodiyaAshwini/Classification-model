using MachineLearning.Common.Interfaces;
using MachineLearning.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace MachineLearning.Common
{
    /// <summary>
    /// Base class for Trainers.
    /// This class exposes methods for training, evaluating and saving ML Models.
    /// Classes that inherit this class need to assing concrete model and name; and to implement data pre-processing.
    /// </summary>
    public abstract class TrainerBase<TParameters>: ITrainerBase
        where TParameters : class
    {
        public string Name { get; protected set; }
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "data.csv");
        protected readonly MLContext mlContext;
        protected DataOperationsCatalog.TrainTestData _dataSplit;
        protected ITrainerEstimator<MulticlassPredictionTransformer<TParameters>, TParameters> _model;
        protected ITransformer _trainedModel;

        protected TrainerBase()
        {
            mlContext = new MLContext(111);
        }

        /// <summary>
        /// Train model on defined data.
        /// </summary>
        /// <param name="data.csv"></param>
        public void Fit(string trainingFileName)
        {
            if (!File.Exists(trainingFileName))
            {
                throw new FileNotFoundException($"File {trainingFileName} doesn't exist.");
            }

            _dataSplit = LoadAndPrepareData(trainingFileName);
            var dataProcessPipeline = BuildDataProcessingPipeline();
            //var trainingPipeline = dataProcessPipeline.Append(_model);
            var trainingPipeline = dataProcessPipeline.Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            _trainedModel = trainingPipeline.Fit(_dataSplit.TrainSet);
        }

        /// <summary>
        /// Evaluate trained model.
        /// </summary>
        /// <returns>Multiclass classification metrics  object.</returns>
        public MulticlassClassificationMetrics Evaluate()
        {
            var testSetTransform = _trainedModel.Transform(_dataSplit.TestSet);

            return mlContext.MulticlassClassification.Evaluate(testSetTransform);
        }

        /// <summary>
        /// Save Model in the file.
        /// </summary>
        public void Save()
        {
            mlContext.Model.Save(_trainedModel, _dataSplit.TrainSet.Schema, ModelPath);
        }

        /// <summary>
        /// Feature engeneering and data pre-processing.
        /// </summary>
        /// <returns>Data Processing Pipeline.</returns>
        private EstimatorChain<ColumnConcatenatingTransformer> BuildDataProcessingPipeline()
        {
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Element", outputColumnName: "Label")
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "ControlId", outputColumnName: "ControlIdFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Name", outputColumnName: "NameFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "CSSClass", outputColumnName: "CSSClassFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Value", outputColumnName: "ValueFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Role", outputColumnName: "RoleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Type", outputColumnName: "TypeFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Title", outputColumnName: "TitleFeaturized"))
                .Append(mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Href", outputColumnName: "HrefFeaturized"))
                .Append(mlContext.Transforms.Concatenate("Features", "ControlIdFeaturized", "NameFeaturized", "CSSClassFeaturized"
                    , "ValueFeaturized", "RoleFeaturized", "TypeFeaturized", "TitleFeaturized", "HrefFeaturized"))
                .AppendCacheCheckpoint(mlContext);

            return dataProcessPipeline;
        }

        private DataOperationsCatalog.TrainTestData LoadAndPrepareData(string trainingFileName)
        {
            var trainingDataView = mlContext.Data.LoadFromTextFile<ElementData>(trainingFileName, hasHeader: true, separatorChar: ',');
            return mlContext.Data.TrainTestSplit(trainingDataView, testFraction: 0.3);
        }
    }
}
