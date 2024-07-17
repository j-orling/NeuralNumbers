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
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                for (int i = 0; i < trainingData.Length; i++)
                {
                    // Pass data forward
                    float[] inputs = trainingData[i];
                    float[] targets = targetData[i];
                    float[] outputs = FeedForward(inputs);

                    // Calculate errors
                    float[] errors = new float[targets.Length];
                    for (int j = 0; j < targets.Length; j++)
                    {
                        errors[j] = targets[j] - outputs[j];
                    }

                    // Backpropagation
                    for (int j = layers.Count - 1; j >= 0; j--)
                    {
                        errors = layers[j].Backpropagate(inputs, errors, learningRate);
                    }
                }

                // Print progress in console
                Console.WriteLine($"Epoch {epoch + 1}/{epochs} completed.");
            }
        }

        // Save entire network
        public void SaveNetwork(string filePath)
        {
            using (StreamWriter writer = new StreamWriter(filePath))
            {
                writer.WriteLine(layers.Count);
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
