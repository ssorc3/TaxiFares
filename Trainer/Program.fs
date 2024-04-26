open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms

let baseDatasetsLocation = @"../Data"
let trainDataPath = $"{baseDatasetsLocation}/taxi-fare-train.csv"
let testDataPath = $"{baseDatasetsLocation}/taxi-fare-test.csv"

let baseModelsPath = @"../Models"
let modelPath = $"{baseModelsPath}/TaxiFareModel.zip"

let downcastPipeline (x : IEstimator<_>) =
    match x with
    | :? IEstimator<ITransformer> as y -> y
    | _ -> failwith "downcastPipeline: expecting an IEstimator<ITransformer>"

let mlContext = new MLContext(seed = 0)
let baseTrainingDataView = mlContext.Data.LoadFromTextFile(trainDataPath, hasHeader = true, separatorChar = ',')
let testDataView = mlContext.Data.LoadFromTextFile(testDataPath, hasHeader = true, separatorChar = ',')

let trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, "FareAmount", lowerBound = 1., upperBound = 150.)

let dataProcessPipeline =
    EstimatorChain()
        .Append(mlContext.Transforms.CopyColumns("Label", "FareAmount"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorIdEncoded", "VendorId"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCodeEncoded", "RateCode"))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentTypeEncoded", "PaymentType"))
        .Append(mlContext.Transforms.NormalizeMeanVariance("PassengerCount", "PassengerCount"))
        .Append(mlContext.Transforms.NormalizeMeanVariance("TripTime", "TripTime"))
        .Append(mlContext.Transforms.NormalizeMeanVariance("TripDistance", "TripDistance"))
        .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", "PassengerCount", "TripTime", "TripDistance"))
        .AppendCacheCheckpoint(mlContext)
        |> downcastPipeline

let trainer = mlContext.Regression.Trainers.Sdca(labelColumnName = "Label", featureColumnName = "Features")
let modelBuilder = dataProcessPipeline.Append trainer

let trainedModel = modelBuilder.Fit trainingDataView

let metrics =
    let predictions = trainedModel.Transform testDataView
    mlContext.Regression.Evaluate(predictions, "Label", "Score")

Common.ConsoleHelper.printRegressionMetrics (trainer.ToString()) metrics

printfn "================= Saving model to file ======================="
let fs = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write)
mlContext.Model.Save(trainedModel, trainingDataView.Schema, fs)
fs.Dispose()
