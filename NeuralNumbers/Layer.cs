using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNumbers
{
    internal class Layer
    {
        int numInputs;
        int numOutputs;
        float bias;
        float[,] weightValues;
        float[,] weightGradients;

        public Layer(int numInputs, int numOutputs)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            weightValues = new float[numInputs, numOutputs];
            weightGradients = new float[numInputs, numOutputs];
            InitValues();
        }

        // Polymorphism - when loading, set values from variables instead of InitValues
        public Layer(int numInputs, int numOutputs, float[,] weights, float bias)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.weightValues = weights;
            this.bias = bias;
        }

        //Initialize weight values
        private void InitValues()
        {
            Random rand = new Random();
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    weightValues[i, j] = (float)(rand.NextDouble() * 2.0 - 1.0); // Initialize weights to random values between -1 and 1
                }
            }
            bias = 0.0f;
        }

        // Helper sigmoid function
        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value * -1);
            return 1 / (1.0f + k);
        }

        // Derivative of sigmoid
        public static float SigmoidDerivative(double value)
        {
            float sigmoid = Sigmoid(value);
            return sigmoid * (1.0f - sigmoid);
        }

        // Prediction - returns the result to forward to the next layer
        public float[] CalculateResult(float[] inputs)
        {
            float[] result = new float[numOutputs];

            for (int j = 0; j < numOutputs; j++)
            {
                float sum = 0.0f;
                for (int i = 0; i < numInputs; i++)
                {
                    sum += inputs[i] * weightValues[i, j];
                }
                sum += bias;
                result[j] = Sigmoid(sum);
            }

            return result;
        }

        // Back propagation
        public float[] Backpropagate(float[] inputs, float[] errors, float learningRate)
        {
            float[] inputErrors = new float[numInputs];

            // Compute gradients
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    float output = Sigmoid(inputs[i] * weightValues[i, j] + bias);
                    float gradient = errors[j] * SigmoidDerivative(output);
                    weightGradients[i, j] = gradient;
                    inputErrors[i] += gradient * weightValues[i, j];
                }
            }

            // Update weights and bias
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    weightValues[i, j] -= learningRate * weightGradients[i, j];
                }
            }

            bias -= learningRate * errors[0]; // Update bias

            return inputErrors;
        }

        // Save layer
        public void Save(StreamWriter sw)
        {
            sw.WriteLine(numInputs);
            sw.WriteLine(numOutputs);
            sw.WriteLine(bias);
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    sw.WriteLine(weightValues[i, j]);
                }
            }
        }

        // Load layer
        public static Layer Load(StreamReader sr)
        {
            int numInputs = int.Parse(sr.ReadLine());
            int numOutputs = int.Parse(sr.ReadLine());
            float bias = float.Parse(sr.ReadLine());
            float[,] weights = new float[numInputs, numOutputs];
            for (int i = 0; i < numInputs; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    weights[i, j] = float.Parse(sr.ReadLine());
                }
            }
            return new Layer(numInputs, numOutputs, weights, bias);
        }
    }
}
