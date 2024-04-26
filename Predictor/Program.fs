open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms
open TaxiFare.DataStructures

let baseModelsPath = @"../Models"
let modelPath = $"{baseModelsPath}/TaxiFareModel.zip"

let mlContext = new MLContext()

let taxiTripSample =
    {
        VendorId = "VTS"
        RateCode = "1"
        PassengerCount = 1.0f
        TripTime = 1140.0f
        TripDistance = 3.75f
        PaymentType = "CRD"
        FareAmount = 0.0f // to predict
    }

let predict =
    let model, inputSchema =
        use s = File.OpenRead(modelPath)
        mlContext.Model.Load(s)
    let predictionFunction = mlContext.Model.CreatePredictionEngine(model)
    predictionFunction.Predict

let stopwatch = System.Diagnostics.Stopwatch.StartNew()
let resultPrediction = predict taxiTripSample
stopwatch.Stop()

printfn "================== Single Prediction ===================="
printfn $"Predicted fare: %.4f{resultPrediction.FareAmount}, actual fare: 15.5"
printfn "========================================================="
printfn $"Prediction time %f{stopwatch.Elapsed.TotalMilliseconds} milliseconds"
