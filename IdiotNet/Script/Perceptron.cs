using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Bgoon;

namespace IdiotNet {
	public class Perceptron {
		public INLayer PrevNode => ownerNode.prevNode;
		public INLayer NextNode => ownerNode.nextNode;
		public INLayer ownerNode;
		public int index;
		public float[] weights;
		public float[] actualWeights;
		public float[] learnWeights;
		public float[] vs;
		public float net;
		public float d;

		public Perceptron(INLayer ownerNode, int index) {
			this.ownerNode = ownerNode;
			this.index = index;
		}
		public void Init(int inputCount) {
			if (PrevNode == null)
				return;

			if(ownerNode.includeBias) {
				weights = new float[inputCount + 1]; //Add Bias
				weights[inputCount] = 0f; //Bias
				vs = new float[weights.Length];
				learnWeights = new float[weights.Length];
				actualWeights = new float[weights.Length];
				for (int i = 0; i < weights.Length - 1; ++i) {
					//Init Weight
					weights[i] = GetRandomValue();
				}
			} else {
				weights = new float[inputCount];
				vs = new float[weights.Length];
				learnWeights = new float[weights.Length];
				actualWeights = new float[weights.Length];
				for (int i = 0; i < weights.Length; ++i) {
					//Init Weight
					weights[i] = GetRandomValue();
				}
			}
		}
		private float GetRandomValue() {
			float inputCount = PrevNode.PerceptCount;
			float outputCount = ownerNode.PerceptCount;

			//return BRandom.Range(-0.1f, 0.1f);

			float stddev = Mathf.Sqrt(3f / (inputCount + outputCount));
			return BRandom.RandomGauss(0f, stddev);

			//float mean = (inputCount + outputCount) / 2f;
			//float sigma = (outputCount - mean) / 3f;
			//return BRandom.RandomGauss(mean, sigma) / Mathf.Sqrt(inputCount * 0.5f);

			return BRandom.RandomGauss(0.01f, 2f / Mathf.Sqrt(inputCount));
		}
	}
}
