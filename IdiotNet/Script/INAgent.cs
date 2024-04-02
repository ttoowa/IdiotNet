using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IdiotNet {
	public class INAgent {

		public int LayerCount => layerList.Count;
		public INLayer OutputLayer => layerList[LayerCount - 1];
		public INLayer InputLayer => layerList[0];
		public float learningRate;
		public string name;
		public readonly ActivationFuncType activationFuncType;
		public readonly List<INLayer> layerList;
		public readonly int batchCount;
		private int batchNum;
		public float BatchFactor {
			get; private set;
		}
		public readonly float dropout;

		public INAgent(ActivationFuncType activationFuncType, float dropout = 0.8f, int batchCount = 10, float learningRate = 0.001f) {
			this.activationFuncType = activationFuncType;
			this.batchCount = batchCount;
			this.learningRate = learningRate;
			this.dropout = dropout;
			layerList = new List<INLayer>();
			name = "ZONE";
			BatchFactor = 1f / batchCount;
		}
		public void AddLayer(int perceptCount) {
			bool includeBias;
			INLayer prevLayer;
			if(layerList.Count > 0) {
				prevLayer = OutputLayer;
				includeBias = true;
			} else {
				prevLayer = null;
				includeBias = false;
			}
			INLayer layer = new INLayer(this, perceptCount, prevLayer, includeBias);
			layerList.Add(layer);
		}
		public void Init() {
			if(layerList.Count > 0) {
				InputLayer.Init();
			}
		}

		public float Calc(params float[] inputs) {
			INLayer inputLayer = InputLayer;
			for(int i=0; i<inputs.Length; ++i) {
				inputLayer.percepts[i].net = inputs[i];
			}
			inputLayer.CalcNode();

			return OutputLayer.percepts[0].net;
		}
		public float CalcLearn(float[] trueOutputs, params float[] inputs) {
			bool isLearnFrame = false;
			if (++batchNum >= batchCount) {
				batchNum = 0;
				isLearnFrame = true;
			}

			float result = Calc(inputs);
			OutputLayer.LearnNode(isLearnFrame, trueOutputs);
			return result;
		}
	}
}
