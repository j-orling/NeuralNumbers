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
            const int trainingStart = 1, trainingEnd = 30000; //30000
            const int validStart = trainingEnd + 1, validEnd = validStart + 9999; //9999
            const int demoStart = validEnd + 1, demoEnd = demoStart + 99;

            string filePath = "C:\\Users\\jorli\\Downloads\\assignment5.csv";
            string networkDataPath = "C:\\Users\\jorli\\source\\repos\\NeuralNumbers\\trained_model.txt";

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

                        // Normalize training data
                        NormalizeData(trainingInputs);
                        
                        // If training, we create a new network - otherwise we load an old one
                        int[] layerSizes = new int[] { trainingInputs[0].Length, 100, 50, 10 };
                        Network network = new Network(layerSizes);

                        network.Train(trainingInputs, trainingTargets, 20, epochs: 10, learningRate: 0.1f);
                        network.trained = true;

                        Console.WriteLine("Do you wish to save this network? (y/n)");
                        string saveAns = Console.ReadLine();

                        if(saveAns == "y")
                        {
                            network.Save(networkDataPath);
                        }

                        break;
                    }
                case "v":
                    {
                        (float[][] trainingInputs, float[][] trainingTargets) = ReadValues(filePath, validStart, validEnd);

                        // Load data from previously trained network
                        Network network = Load(networkDataPath);
                        network.ValidateData(trainingInputs, trainingTargets);
                        Console.ReadLine();

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

        // Load entire network, and set all individual layer parameters
        public static Network Load(string filePath)
        {
            using (StreamReader reader = new StreamReader(filePath))
            {
                int layerCount = int.Parse(reader.ReadLine());
                List<Layer> layers = new List<Layer>();
                for (int i = 0; i < layerCount; i++)
                {
                    layers.Add(Layer.Load(reader));
                }
                Network network = new Network(new int[0]);
                network.layers = layers;
                reader.Close();
                return network;
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
                        string[] values = line.Split(',');

                        float[] targets = new float[10];
                        targets[int.Parse(values.First())] = 1.0f;
                        List<string> cleaned = values.ToList();
                        cleaned.RemoveAt(0);
                        values = cleaned.ToArray();
                        float[] inputs = values.Take(values.Length).Select(float.Parse).ToArray();

                        inputValues.Add(inputs);
                        targetValues.Add(targets);
                    }
                    lineNum++;
                    
                }

                sr.Close();
            }

            return (inputValues.ToArray(), targetValues.ToArray());
        }

        // "Squish" data so the numbers are smaller and thus have less overall impact on the network
        static void NormalizeData(float[][] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                float max = data[i].Max();
                float min = data[i].Min();
                for (int j = 0; j < data[i].Length; j++)
                {
                    data[i][j] = (data[i][j] - min) / (max - min);
                }
            }
        }
    }
}
