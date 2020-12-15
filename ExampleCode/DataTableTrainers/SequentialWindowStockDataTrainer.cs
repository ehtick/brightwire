﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using BrightData;
using BrightData.FloatTensors;
using BrightTable;
using BrightWire;
using BrightWire.Models;

namespace ExampleCode.DataTableTrainers
{
    class SequentialWindowStockDataTrainer : DataTableTrainer
    {
        public SequentialWindowStockDataTrainer(IRowOrientedDataTable table) : base(table)
        {
        }

        public void TrainLSTM(uint hiddenLayerSize)
        {
            var graph = Table.Context.CreateGraphFactory();
            var errorMetric = graph.ErrorMetric.Quadratic;

            // create the property set
            graph.CurrentPropertySet
                .Use(graph.GradientDescent.Adam)
                .Use(graph.WeightInitialisation.Xavier);

            // create the engine
            var trainingData = graph.CreateDataSource(Training);
            var testData = trainingData.CloneWith(Test);
            var engine = graph.CreateTrainingEngine(trainingData, learningRate: 0.03f, batchSize: 128);

            // build the network
            graph.Connect(engine)
                .AddLstm(hiddenLayerSize)
                .AddFeedForward(engine.DataSource.GetOutputSizeOrThrow())
                .Add(graph.TanhActivation())
                .AddBackpropagationThroughTime(errorMetric);

            // train the network and restore the best result
            GraphModel bestNetwork = null;
            engine.Train(5, testData, errorMetric, model => bestNetwork = model);
            if (bestNetwork != null) {
                // execute each row of the test data on an execution engine
                var executionEngine = graph.CreateEngine(bestNetwork.Graph);
                var results = executionEngine.Execute(testData).OrderSequentialOutput();
                var expectedOutput = Test.Column<Vector<float>>(1).ToArray();

                var score = results.Select((r, i) => errorMetric.Compute(r.Last(), expectedOutput[i])).Average();
                Console.WriteLine(score);


            }
        }
    }
}