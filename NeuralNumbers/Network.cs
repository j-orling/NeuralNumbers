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
            float[][] rawOutputs = new float[layers.Count][];
            for(int i = 0; i < layers.Count; i++)
            {
                outputs[i] = new float[layers[i].numOutputs];
                rawOutputs[i] = new float[layers[i].numOutputs];
            }

            (outputs[0], rawOutputs[0]) = layers[0].CalculateResult(inputs);
            for (int i = 1; i < layers.Count; i++)
            {
                (outputs[i], rawOutputs[i]) = layers[i].CalculateResult(outputs[i - 1]);
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
                for (int i = 0; i < layers.Count; i++)
                {
                    deltaB[i] = new float[layers[i].numOutputs];
                    deltaW[i] = new float[layers[i].numOutputs][];
                    for(int j = 0; j < layers[i].numOutputs; j++)
                    {
                        deltaW[i][j] = new float[layers[i].numInputs];
                    }
                }
                
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
                        delta[j] = (outputs.Last()[j] - targets[j]) * SigmoidPrime(rawOutputs.Last()[j]);
                        
                    }

                    if (outputs.Last().ToList().IndexOf(outputs.Last().Max()) == targets.ToList().IndexOf(targets.Max()))
                    {
                        correct++;
                    }

                    // Backpropagation
                    float[][] nablaB = new float[layers.Count][];
                    float[][][] nablaW = new float[layers.Count][][];
                    for(int j = 0; j < layers.Count; j++)
                    {
                        nablaB[j] = new float[layers[j].bias.Length];
                        nablaW[j] = new float[layers[j].numOutputs][];
                        for(int k = 0; k < layers[j].numOutputs; k++)
                        {
                            nablaW[j][k] = new float[layers[j].numInputs];
                        }
                    }
                    
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
                        nablaW[j] = DotProduct(delta, j > 0 ? outputs[j - 1] : layers[0].CalculateResult(inputs).Item1);
                    }

                    // Accumulate bias differences
                    for(int j = 0; j < nablaB.Length; j++)
                    {
                        for(int k = 0; k < nablaB[j].Length; k++)
                        {
                            deltaB[j][k] += nablaB[j][k];
                        }
                    }

                    // Accumulate weight differences
                    for(int j = 0; j < nablaW.Length; j++)
                    {
                        for(int k = 0; k < nablaW[j].Length; k++)
                        {
                            for(int l = 0; l < nablaW[j][k].Length; l++)
                            {
                                deltaW[j][k][l] += nablaW[j][k][l];
                            }
                        }
                    }

                    Console.WriteLine("Output: " + outputs.Last().Last());
                    Console.WriteLine("Target: " + targets.Last());
                }

                // Update bias and weight values
                for(int i = 0; i < layers.Count; i++)
                {
                    for(int j = 0; j < layers[i].bias.Length; j++)
                    {
                        layers[i].bias[j] += deltaB[i][j];
                    }

                    for(int j = 0; j < layers[i].weightValues.Length; j++)
                    {
                        for(int k = 0; k < layers[i].weightValues[j].Length; k++)
                        {
                            layers[i].weightValues[j][k] += deltaW[i][j][k];
                        }
                    }
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
            return Sigmoid(value) * (1.0f - Sigmoid(value));
        }

        private float Sigmoid(float value)
        {
            return 1 / (float)(1.0f + Math.Exp(-1 * value));
        }

        private float[][] DotProduct(float[] vector1, float[] vector2)
        {
            float[][] res = new float[vector2.Length][];
            for(int i = 0; i < vector2.Length; i++)
            {
                res[i] = new float[vector1.Length];
            }
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
