using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNumbers
{
    internal class Layer
    {
        public int numInputs;
        public float[] inputs;
        public int numOutputs;
        public float[] outputs;
        public float[] bias;
        public float[][] weightValues;

        public Layer(int numInputs, int numOutputs)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            weightValues = new float[numOutputs][];
            for(int i = 0; i < numOutputs; i++)
            {
                weightValues[i] = new float[numInputs];
            }
            bias = new float[numOutputs];
            InitValues();
        }

        // Polymorphism - when loading, set values from variables instead of InitValues
        public Layer(int numInputs, int numOutputs, float[][] weights, float[] bias)
        {
            this.numInputs = numInputs;
            this.numOutputs = numOutputs;
            this.weightValues = weights;
            this.bias = bias;
        }

        //Initialize weight + bias values
        private void InitValues()
        {
            Random rand = new Random();
            for (int i = 0; i < numOutputs; i++)
            {
                bias[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
                for (int j = 0; j < numInputs; j++)
                {
                    weightValues[i][j] = (float)(rand.NextDouble() * 2.0 - 1.0);
                }
            }
        }

        // Sigmoid activation function
        public static float Sigmoid(double value)
        {
            float k = (float)Math.Exp(value * -1);
            return 1 / (1.0f + k);
        }

        // Prediction - returns the result to forward to the next layer
        public float[] CalculateResult(float[] inputs)
        {
            this.inputs = inputs;
            float[] result = new float[numOutputs];

            for (int j = 0; j < numOutputs; j++)
            {
                float sum = 0.0f;
                for (int i = 0; i < numInputs; i++)
                {
                    sum += inputs[i] * weightValues[j][i];
                }
                sum += bias[j];
                result[j] = Sigmoid(sum);
            }

            // Stored for backprop
            this.outputs = result;
            return result;
        }

        // Save layer
        public void Save(StreamWriter sw)
        {
            sw.WriteLine(numInputs);
            sw.WriteLine(numOutputs);
            for (int i = 0; i < numOutputs; i++)
            {
                sw.WriteLine(bias[i]);
                for (int j = 0; j < numInputs; j++)
                {
                    sw.WriteLine(weightValues[i][j]);
                }
            }
        }

        // Load layer
        public static Layer Load(StreamReader sr)
        {
            int numInputs = int.Parse(sr.ReadLine());
            int numOutputs = int.Parse(sr.ReadLine());
            float[] bias = new float[numOutputs];
            float[][] weights = new float[numOutputs][];
            for (int i = 0; i < numOutputs; i++)
            {
                weights[i] = new float[numInputs];
                bias[i] = float.Parse(sr.ReadLine());
                for (int j = 0; j < numInputs; j++)
                {
                    weights[i][j] = float.Parse(sr.ReadLine());
                }
            }
            return new Layer(numInputs, numOutputs, weights, bias);
        }
    }
}
