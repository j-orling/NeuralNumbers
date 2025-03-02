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
        public List<Layer> layers = new List<Layer>();
        public int largestLayer = 1;
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

        public float[][] FeedForward(float[] inputs)
        {
            float[][] outputs = new float[layers.Count][];
            for (int i = 0; i < layers.Count; i++)
            {
                outputs[i] = new float[layers[i].numOutputs];
            }

            outputs[0] = layers[0].CalculateResult(inputs);
            for (int i = 1; i < layers.Count; i++)
            {
                outputs[i] = layers[i].CalculateResult(outputs[i - 1]);
            }

            return outputs;
        }

        public void Backpropagate(float[] target, float learningRate)
        {
            // Calculate output layer deltas
            float[][] deltas = new float[layers.Count][];
            for (int i = layers.Count - 1; i >= 0; i--)
            {
                deltas[i] = new float[layers[i].numOutputs];
                for (int j = 0; j < layers[i].numOutputs; j++)
                {
                    if (i == layers.Count - 1) // Output layer
                    {
                        float output = layers[i].outputs[j];
                        deltas[i][j] = (output - target[j]) * output * (1 - output);
                    }
                    else // Hidden layers
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < layers[i + 1].numOutputs; k++)
                        {
                            sum += deltas[i + 1][k] * layers[i + 1].weightValues[k][j];
                        }
                        float output = layers[i].outputs[j];
                        deltas[i][j] = sum * output * (1 - output);
                    }
                }
            }

            // Update weights and biases
            for (int i = 0; i < layers.Count; i++)
            {
                for (int j = 0; j < layers[i].numOutputs; j++)
                {
                    layers[i].bias[j] -= learningRate * deltas[i][j];
                    for (int k = 0; k < layers[i].numInputs; k++)
                    {
                        layers[i].weightValues[j][k] -= learningRate * deltas[i][j] * layers[i].inputs[k];
                    }
                }
            }
        }

        public float Train(float[][] inputData, float[][] targData, int batchSize, int epochs, float learningRate)
        {
            Console.WriteLine("Starting training");

            int correct = 0;
            float lastBatchErr = 0.0f;
            float sumError = 0.0f;
            float initLearningRate = learningRate;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                sumError = 0.0f;
                correct = 0;

                for (int i = 0; i < inputData.Length; i += batchSize)
                {
                    int end = Math.Min(i + batchSize, inputData.Length);
                    for (int j = i; j < end; j++)
                    {
                        float[][] outputs = FeedForward(inputData[j]);
                        float[] output = outputs.Last();
                        float[] target = targData[j];

                        // Calculate error and backpropagate
                        float error = 0.0f;
                        for (int k = 0; k < output.Length; k++)
                        {
                            float delta = target[k] - output[k];
                            error += delta * delta;
                        }
                        sumError += error;

                        // Backpropagation
                        Backpropagate(target, learningRate);

                        // Check if the prediction is correct
                        if (Array.IndexOf(output, output.Max()) == Array.IndexOf(target, target.Max()))
                        {
                            correct++;
                        }
                    }
                }

                // Adjust learning rate if using a learning rate schedule
                learningRate = initLearningRate * (1.0f / (1.0f + 0.01f * epoch));

                Console.WriteLine($"Epoch {epoch + 1}/{epochs}, Error: {sumError}, Accuracy: {(float)correct / inputData.Length}");
            }

            return sumError;
        }

        // Check if the network achieves at least 90% accuracy
        public bool CheckAccuracy(float[][] inputs, float[][] targets)
        {
            int correct = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                float[][] outputs = FeedForward(inputs[i]);
                if (Array.IndexOf(outputs.Last(), outputs.Last().Max()) == Array.IndexOf(targets[i], 1))
                {
                    correct++;
                }
            }
            double accuracy = (double)correct / inputs.Length;
            return accuracy >= 0.90;
        }

        private (float[][], float[][]) ShuffleData(float[][] trainingData, float[][] targetData)
        {
            float[][] training = new float[trainingData.Length][];
            float[][] target = new float[targetData.Length][];

            Random rng = new Random();
            int i = trainingData.Length;
            while (i > 1)
            {
                i--;
                int j = rng.Next(i);
                float[] inputVals = trainingData[j];
                float[] targetVals = targetData[j];
                trainingData[j] = trainingData[i];
                targetData[j] = targetData[i];
                trainingData[i] = inputVals;
                targetData[i] = targetVals;
            }

            return (trainingData, targetData);
        }

        public void ValidateData(float[][] trainingData, float[][] targetData)
        {
            int correct = 0;

            for (int i = 0; i < trainingData.Length; i++)
            {
                float[] inputs = trainingData[i];
                float[] targets = targetData[i];

                float[][] outputs = FeedForward(inputs);

                // Check if correct and print debug info
                correct = Array.IndexOf(outputs.Last(), outputs.Last().Max()) == Array.IndexOf(targets, 1) ? correct + 1 : correct;
                Console.WriteLine("Target value: " + Array.IndexOf(targets, 1));
                Console.WriteLine("Output value: " + Array.IndexOf(outputs.Last(), outputs.Last().Max()));
                Console.WriteLine("-------------------------------------------");
            }

            double correctProc = ((double)correct / ((double)trainingData.Length));

            Console.WriteLine("Correct: " + correctProc.ToString("P2"));
        }

        private float SigmoidPrime(float value)
        {
            float sigVal = Sigmoid(value);
            float ret = Sigmoid(value) * (1.0f - Sigmoid(value));
            return ret;
        }

        private float Sigmoid(float value)
        {
            return 1 / (float)(1.0f + Math.Exp(-1 * value));
        }

        private float[][] DotProduct(float[] vector1, float[] vector2)
        {
            float[][] res = new float[vector1.Length][];
            for (int i = 0; i < vector1.Length; i++)
            {
                res[i] = new float[vector2.Length];
            }
            for (int i = 0; i < vector1.Length; i++)
            {
                for (int j = 0; j < vector2.Length; j++)
                {
                    res[i][j] += vector1[i] * vector2[j];
                }
            }
            return res;
        }

        // Override in case of two-dimensional matrix
        private float[] DotProduct(float[][] vector1, float[] vector2)
        {
            float[] res = new float[vector1[0].Length];
            for (int i = 0; i < vector1[0].Length; i++)
            {
                res[i] = 0;
                for (int j = 0; j < vector2.Length; j++)
                {
                    res[i] += vector1[j][i] * vector2[j];
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
                foreach (Layer layer in layers)
                {
                    layer.Save(writer);
                }

                writer.Close();
            }
        }
    }


}