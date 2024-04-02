using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Bgoon;
using Bgoon.MultiThread;

namespace IdiotNet {
	public class INLayer {
		public float LearningRate => ownerAgent.learningRate;
		public int PerceptCount => percepts.Length;
		public readonly INAgent ownerAgent;
		public Perceptron[] percepts;
		public INLayer prevNode;
		public INLayer nextNode;
		public bool includeBias;
		private int inputCount;
		private ActivationFuncType activationFuncType;
		private delegate float ActivationFunc(float input);
		private ActivationFunc activeFunc;
		private ActivationFunc dActiveFunc;

		public INLayer(INAgent ownerAgent, int length, INLayer prevNode, bool includeBias = true) {
			this.ownerAgent = ownerAgent;
			this.prevNode = prevNode;
			this.includeBias = includeBias;

			if (prevNode != null) {
				prevNode.ConnectNext(this);
				inputCount = prevNode.PerceptCount;
			}

			percepts = new Perceptron[length];

			for (int i = 0; i < percepts.Length; ++i) {
				percepts[i] = new Perceptron(this, i);
			}

			SetActivationFunc(ownerAgent.activationFuncType);
		}
		private void ConnectNext(INLayer nextNode) {
			this.nextNode = nextNode;
			nextNode.prevNode = this;
		}

		/// <summary>
		/// 첫 번째 레이어에서 호출하세요
		/// </summary>
		public void Init() {
			for (int i = 0; i < percepts.Length; ++i) {
				percepts[i].Init(inputCount);
			}

			if (nextNode != null) {
				nextNode.Init();
			}
		}
		/// <summary>
		/// 첫 번째 레이어에서 호출하세요
		/// </summary>
		public void CalcNode() {
			bool useDropout = (prevNode != null && nextNode != null) && (PerceptCount > ownerAgent.dropout * 10f);
			if (prevNode != null) {
				//입력 노드가 아니면
				for (int cpI = 0; cpI < percepts.Length; ++cpI) {
					Perceptron percept = percepts[cpI];

					if (includeBias) {
						percept.net = percept.weights[percept.weights.Length - 1]; //Bias
					} else {
						percept.net = 0f;
					}
					for (int ppI = 0; ppI < prevNode.percepts.Length; ++ppI) {
						Perceptron prevPercep = prevNode.percepts[ppI];

						if (useDropout && (BRandom.Value > ownerAgent.dropout)) {
							percept.net += prevPercep.net * percept.weights[ppI];
							percept.actualWeights[ppI] = percept.weights[ppI];
						} else {
							percept.actualWeights[ppI] = 0f;
						}
					}
					percept.net = activeFunc(percept.net);
				}
			}
			if (nextNode != null) {
				//출력 노드가 아니면 다음 레이어 연산
				nextNode.CalcNode();
			}
		}
		/// <summary>
		/// 마지막 레이어에서 호출하세요
		/// </summary>
		/// <param name="trueOutput"></param>
		public void LearnNode(bool isLearnFrame, params float[] trueOutputs) {
			for (int cpI = 0; cpI < PerceptCount; ++cpI) {
				Perceptron cPercept = percepts[cpI];

				//cPercept.d = BMath.SigmoidDifferential(2f / PerceptCount * (cPercept.net - trueOutput));
				cPercept.d = 2f / PerceptCount * (cPercept.net - trueOutputs[cpI]);
				cPercept.d *= dActiveFunc(cPercept.net);

			}
			if (prevNode != null) {
				prevNode.LearnRecursion(isLearnFrame);
			}
		}
		private void LearnRecursion(bool isLearnFrame) {

			for (int cpI = 0; cpI < PerceptCount; ++cpI) {
				Perceptron cPercept = percepts[cpI];
				cPercept.d = 0f;
				for (int npI = 0; npI < nextNode.PerceptCount; ++npI) {
					Perceptron nPercept = nextNode.percepts[npI];

					cPercept.d += nPercept.d * nPercept.actualWeights[cpI];
					//nPercept.weights[cpI] -= (cPercept.net * nPercept.d) * LearningRate;
					nPercept.learnWeights[cpI] += Momentum(nPercept, cpI, nPercept.d) * LearningRate;
					if(isLearnFrame) {
						nPercept.weights[cpI] -= nPercept.learnWeights[cpI] * ownerAgent.BatchFactor;
						nPercept.learnWeights[cpI] = 0f;
					}
				}
				cPercept.d *= dActiveFunc(cPercept.net);
			}

			if (nextNode.includeBias) {
				int cpI = percepts.Length;
				for (int npI = 0; npI < nextNode.PerceptCount; ++npI) {
					Perceptron nPercept = nextNode.percepts[npI];

					//nPercept.weights[nPercept.weights.Length - 1] -= nPercept.d * LearningRate;
					
					nPercept.learnWeights[cpI] += Momentum(nPercept, cpI, nPercept.d) * LearningRate;
					if (isLearnFrame) {
						nPercept.weights[cpI] -= nPercept.learnWeights[cpI] * ownerAgent.BatchFactor;
						nPercept.learnWeights[cpI] = 0f;
					}
				}
			}

			if (prevNode != null) {
				prevNode.LearnRecursion(isLearnFrame);
			}
		}

		public void SetActivationFunc(ActivationFuncType type) {
			switch (type) {
				case ActivationFuncType.Sigmoid:
					activeFunc = Sigmoid;
					dActiveFunc = DSigmoid;
					break;
				case ActivationFuncType.ReLU:
					activeFunc = ReLU;
					dActiveFunc = DReLU;
					break;
				case ActivationFuncType.LeakyReLU:
					activeFunc = LeakyReLU;
					dActiveFunc = DLeakyReLU;
					break;
			}
		}

		//Activation Functions
		private float Sigmoid(float input) {
			return BMath.Sigmoid(input);
		}
		private float DSigmoid(float input) {
			return input * (1f - input);
		}
		private float ReLU(float input) {
			return Mathf.Max(0f, input);
		}
		private float DReLU(float input) {
			return input > 0 ? 1f : 0f;
		}
		private float LeakyReLU(float input) {
			return input > 0 ? input : input * 0.1f;
		}
		private float DLeakyReLU(float input) {
			return input > 0 ? 1f : 0.1f;
		}

		//Slope Descent Method
		private float Momentum(Perceptron percept, int wIndex, float dW) {
			//return dW;
			const float Term = 0.98f;

			percept.vs[wIndex] = percept.vs[wIndex] * Term + dW;
			return dW + percept.vs[wIndex];
		}
	}
}
