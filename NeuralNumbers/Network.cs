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
        }

        public float[] FeedForward(float[] inputs)
        {
            float[] outputs = inputs;

            foreach (Layer layer in layers)
            {
                outputs = layer.CalculateResult(outputs);
            }

            return outputs;
        }

        public void Train(float[][] trainingData, float[][] targetData, int epochs, float learningRate)
        {
            Console.WriteLine("Starting training");
            int correct = 0;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                float totErrors = 0.0f;
                for (int i = 0; i < trainingData.Length; i++)
                {
                    // Pass data forward
                    float[] inputs = trainingData[i];
                    float[] targets = targetData[i];
                    float[] outputs = FeedForward(inputs);

                    // Calculate errors
                    float[] errors = new float[targets.Length];
                    float largestTarget = float.MinValue;
                    int largestTargetIndex = -1;
                    float largestOutput = float.MinValue;
                    int largestOutputIndex = -1;
                    for (int j = 0; j < targets.Length; j++)
                    {
                        errors[j] = targets[j] - outputs[j];

                        if (targets[j] > largestTarget)
                        {
                            largestTarget = targets[j];
                            largestTargetIndex = j;
                        }

                        if (outputs[j] > largestOutput)
                        {
                            largestOutput = outputs[j];
                            largestOutputIndex = j;
                        }

                        totErrors += Math.Abs(errors[j]);
                    }

                    if(largestOutputIndex == largestTargetIndex)
                    {
                        correct++;
                    }

                    // Backpropagation
                    for (int j = layers.Count - 1; j >= 0; j--)
                    {
                        errors = layers[j].Backpropagate(inputs, errors, learningRate);
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
