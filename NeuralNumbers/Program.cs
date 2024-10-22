using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace NeuralNumbers
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Dataset start and end ranges
            const int trainingStart = 1, trainingEnd = 30; //30000
            const int validStart = trainingEnd + 1, validEnd = validStart + 9999;
            const int demoStart = validEnd + 1, demoEnd = demoStart + 99;

            string filePath = "C:\\Users\\jorli\\Downloads\\assignment5.csv";

            // Menu
            Console.WriteLine("t - train");
            Console.WriteLine("v - validate");
            Console.WriteLine("d - demo");
            Console.WriteLine("e - exit");
            string ans = Console.ReadLine();
            

            switch (ans) 
            {
                case "t":
                    {
                        (float[][] trainingInputs, float[][] trainingTargets) = ReadValues(filePath, trainingStart, trainingEnd);
                        // If training, we create a new network - otherwise we load an old one
                        int[] layerSizes = new int[] { trainingInputs[0].Length, 30, 10 };
                        Network network = new Network(layerSizes);

                        network.Train(trainingInputs, trainingTargets, 10, epochs: 5, learningRate: 0.001f);
                        network.trained = true;

                        network.Save("C:\\Users\\jorli\\source\\repos\\NeuralNumbers\\trained_model.txt");
                        break;
                    }
                case "v":
                    {
                        (float[][] validInputs, float[][] validTargets) = ReadValues(filePath, validStart, validEnd);
                        Network network = new Network();

                        break;
                    }
                case "d":
                    {
                        (float[][] validInputs, float[][] validTargets) = ReadValues(filePath, demoStart, demoEnd);
                        break;
                    }
                case "e":
                    {

                        break;
                    }
            }

            

            
        }

        // Get dataset values from csv file
        static (float[][] inputValues, float[][] targetValues) ReadValues(string filename, int startLine, int endLine)
        {
            List<float[]> inputValues = new List<float[]>();
            List<float[]> targetValues = new List<float[]>();

            using (var sr = new StreamReader(filename))
            {
                int lineNum = 0;
                while(!sr.EndOfStream)
                {
                    if(lineNum < startLine || lineNum > endLine)
                    {
                        var line = sr.ReadLine();
                    }
                    else if(lineNum >= startLine && lineNum <= endLine)
                    {
                        var line = sr.ReadLine();
                        var values = line.Split(',');

                        float[] targets = new float[10];
                        targets[int.Parse(values.First())] = 1.0f;
                        float[] inputs = values.Take(values.Length - 1).Select(float.Parse).ToArray();

                        inputValues.Add(inputs);
                        targetValues.Add(targets);
                    }
                    lineNum++;
                    
                }
            }

            // Shuffle values for later minibatching
            Random rng = new Random();
            int i = inputValues.Count;
            while( i > 1)
            {
                i--;
                int j = rng.Next(i + 1);
                float[] inputVals = inputValues[j];
                float[] targetVals = targetValues[j];
                inputValues[j] = inputValues[i];
                targetValues[j] = targetValues[i];
                inputValues[i] = inputVals;
                targetValues[i] = targetVals;
            }

            return (inputValues.ToArray(), targetValues.ToArray());
        }
    }
}
