using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNumbers
{
    internal class Network
    {
        List<Layer> layers = new List<Layer>();
        int largestLayer = 1;
        public bool trained = false;

        public Network()
        {
            
        }

        public Network(int[] layerSizes)
        {
            for (int i = 0; i < layerSizes.Length - 1; i++)
            {
                layers.Add(new Layer(layerSizes[i], layerSizes[i + 1]));
            }

            largestLayer = layerSizes.Max();
        }

        public (float[][], float[][]) FeedForward(float[] inputs)
        {
            float[][] outputs = new float[layers.Count][];
            outputs[0] = inputs;
            float[][] rawOutputs = new float[layers.Count][];

            for (int i = 0; i < layers.Count; i++)
            {
                (outputs[i + 1], rawOutputs[i + 1]) = layers[i].CalculateResult(outputs[i]);
            }

            return (outputs, rawOutputs);
        }

        public void Train(float[][] trainingData, float[][] targetData, int batchSize, int epochs, float learningRate)
        {
            Console.WriteLine("Starting training");

            int correct = 0;
            // Indexing for mini-batching
            int trainingStart = 0;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float totErrors = 0.0f;
                float[][] deltaB = new float[layers.Count][];
                float[][][] deltaW = new float[layers.Count][][];
                for (int i = trainingStart; i < trainingStart + batchSize; i++)
                {
                    // Pass data forward
                    float[] inputs = trainingData[i];
                    float[] targets = targetData[i];
                    // Outputs and rawOutputs store activation values for each layer in the network
                    (float[][] outputs, float[][] rawOutputs) = FeedForward(inputs);

                    // Calculate errors, store in delta
                    float[] delta = new float[targets.Length];
                    for (int j = 0; j < targets.Length; j++)
                    {
                        delta[j] = outputs.Last()[j] - targets[j] * SigmoidPrime(rawOutputs.Last()[j]);
                    }

                    // Backpropagation
                    float[][] nablaB = new float[layers.Count][];
                    float[][][] nablaW = new float[layers.Count][][];
                    nablaB[nablaB.Length - 1] = delta;
                    nablaW[layers.Count - 1] = DotProduct(outputs.Last(), delta);

                    // Start at next to last layer, as we already have computed the last layer in the last loop
                    for (int j = layers.Count - 2; j >= 0; j--)
                    {
                        float[] z = rawOutputs[j];
                        float[] sp = new float[layers[j].numOutputs];
                        for(int k = 0; k < layers[j].numOutputs; k++)
                        {
                            sp[k] = SigmoidPrime(rawOutputs[j][k]);
                        }
                        delta = DotProduct(layers[j + 1].weightValues, delta);
                        nablaB[j] = delta;
                        nablaW[j] = DotProduct(delta, outputs[j - 1]);
                    }

                    // Next step: accumulate bias and weight differences, and then update bias and weight values after each batch

                }

                // Print progress in console
                Console.WriteLine($"Epoch {epoch + 1}/{epochs} completed.");
                Console.WriteLine("Error total: " + totErrors);
                totErrors = 0.0f;

            }
            Console.WriteLine("Ending training");
            Console.WriteLine("Correct: " + correct / trainingData.Length + "%");
        }

        private float SigmoidPrime(float value)
        {
            return value * (1.0f - value);
        }

        private float[][] DotProduct(float[] vector1, float[] vector2)
        {
            float[][] res = new float[vector2.Length][];
            for(int i = 0; i < vector1.Length; i++)
            {
                for(int j = 0; j < vector1.Length; j++)
                {
                    res[i][j] += vector1[i] * vector2[j];
                }
            }
            return res;
        }

        // Override in case of two-dimensional matrix
        private float[] DotProduct(float[][] vector1, float[] vector2)
        {
            float[] res = new float[vector1.Length];
            for(int i = 0; i < vector1.Length; i++)
            {
                for(int j = 0; j < vector2.Length; j++)
                {
                    float intermittentResult = 0;
                    for(int k = 0; k < vector1[i].Length; k++)
                    {
                        intermittentResult += vector1[i][k] * vector2[j];
                    }
                    res[i] += intermittentResult;
                }
            }
            return res;
        }

        // Save entire network
        public void Save(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(layers.Count);
                writer.WriteLine(trained);
                foreach (Layer layer in layers)
                {
                    layer.Save(writer);
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
                return network;
            }
        }
    }
}
