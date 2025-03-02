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
            const int trainingStart = 1, trainingEnd = 10000; //30000
            const int validStart = trainingEnd + 1, validEnd = validStart + 9999; //9999
            const int demoStart = validEnd + 1, demoEnd = demoStart + 99;

            string filePath = "C:\\Users\\jorli\\Downloads\\assignment5.csv";
            string networkDataPath = "C:\\Users\\jorli\\source\\repos\\NeuralNumbers\\trained_model.txt";
            string debugPath = "C:\\Users\\jorli\\source\\repos\\NeuralNumbers\\debug_model.txt";

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
                        int[] layerSizes = new int[] { trainingInputs[0].Length, 50, 40, 10 };
                        Network network = new Network(layerSizes);

                        network.Train(trainingInputs, trainingTargets, 20, epochs: 40, learningRate: 2f);
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
                case "debug":
                    {
                        (float[][] trainingInputs, float[][] trainingTargets) = ReadValues(filePath, trainingStart, validEnd);

                        // If training, we create a new network - otherwise we load an old one
                        int[] layerSizes = new int[] { trainingInputs[0].Length, 30, 30, 10 };
                        Network network = new Network(layerSizes);

                        network.Train(trainingInputs, trainingTargets, 30, epochs: 800, learningRate: 2f);

                        network.Save(debugPath);

                        Network testNet = Load(debugPath);

                        int debugLen = 30;
                        int correct = 0;
                        int loadedCorrect = 0;
                        for(int i = 0; i < debugLen; i++)
                        {
                            (float[][] outputData, float[][] rawData) = network.FeedForward(trainingInputs[i]);
                            (float[][] debugOutput, float[][] debugRaw) = testNet.FeedForward(trainingInputs[i]);

                            correct = Array.IndexOf(outputData.Last(), outputData.Last().Max()) == Array.IndexOf(trainingTargets[i], 1) ? correct + 1 : correct;
                            loadedCorrect = Array.IndexOf(debugOutput.Last(), debugOutput.Last().Max()) == Array.IndexOf(trainingTargets[i], 1) ? loadedCorrect + 1 : loadedCorrect;

                            Console.WriteLine("Saved: " + Array.IndexOf(outputData.Last(), outputData.Last().Max()) + " | Loaded: " + Array.IndexOf(debugOutput.Last(), debugOutput.Last().Max()) + " | Target: " + Array.IndexOf(trainingTargets[i], 1));
                        }

                        double correctProc = (double)correct / (double)debugLen;
                        double correctProcLoaded = (double)loadedCorrect / (double)debugLen;

                        Console.WriteLine("Correct: " + correctProc.ToString("P2"));
                        Console.WriteLine("Correct (loaded): " + correctProcLoaded.ToString("P2"));

                        Console.ReadLine();

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
    }
}
