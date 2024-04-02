using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Bgoon;

namespace IdiotNet {
	public class Program {
		static void Main(string[] args) {

			//LearnMath();
			LearnText();

			Console.ReadLine();
		}
		private static void LearnMath() {
			INAgent agent = new INAgent(ActivationFuncType.LeakyReLU, 0.7f, 30, 0.005f);
			agent.AddLayer(2);
			agent.AddLayer(10);
			agent.AddLayer(10);
			agent.AddLayer(10);
			//agent.AddLayer(60);
			//agent.AddLayer(100);
			//agent.AddLayer(100);
			agent.AddLayer(1);
			//agent.OutputLayer.SetActivationFunc(ActivationFuncType.ReLU);
			agent.Init();

			for (int i = 0; i < 1000000; ++i) {

				bool logFlag = false;
				if (i % 10000 == 0) {
					Console.WriteLine();
					Console.WriteLine(i + "회차");
					logFlag = true;
				}
				float a = BRandom.Value;
				float b = BRandom.Value;
				float trueResult = Calc(a, b);
				float result = agent.CalcLearn(new float[] { trueResult }, a, b);
				if (logFlag) {
					Console.WriteLine("(" + a.ToString("0.00") + "," + b.ToString("0.00") + ") = " + result.ToString("0.00") +
						"… 오차 " + (trueResult - result).ToString("0.000"));
				}
			}
			float Calc(float a, float b) {
				return a*b;
			}
		}
		private const int MaxChar = 122;
		private static void LearnText() {
			const int InputCount = 5;
			const int BatchCount = 2;
			INAgent agent = new INAgent(ActivationFuncType.LeakyReLU, 0.9f, BatchCount, 0.001f);
			agent.AddLayer(InputCount);
			agent.AddLayer(10);
			//agent.AddLayer(10);
			//agent.AddLayer(10);
			//agent.AddLayer(128);
			//agent.AddLayer(300);
			//agent.AddLayer(128);
			agent.AddLayer(1);
			//agent.OutputLayer.SetActivationFunc(ActivationFuncType.Sigmoid);
			agent.Init();
			string learningText = "X:/LearningText.txt".LoadText();

			for (int i = 0; i < 1000000; ++i) {
				if (i % 10000 == 0) {
					Console.WriteLine();
					Console.WriteLine(i + "회차");
					Console.WriteLine(CreateText());
				}
				for (int tI=InputCount; tI<learningText.Length; ++tI) {
					//float[] to = new float[60];
					//to[Mathf.Clamp((int)learningText[tI], 0, 121)] = 1f;
					float[] to = new float[] { GetValueFromChar(learningText[tI]) };
					float[] inputs = new float[InputCount];
					for(int i2=0; i2 < InputCount; ++i2) {
						inputs[i2] = GetValueFromChar(learningText[tI - (InputCount - i2)]);
					}
					float result = agent.CalcLearn(to, inputs);
				}
			}
			string CreateText() {
				List<char> charList = new List<char>("fuck ");

				for (int tI = charList.Count; tI < 300; ++tI) {
					float[] inputs = new float[InputCount];
					for(int i=0; i<InputCount; ++i) {
						inputs[i] = GetValueFromChar(charList[tI - (InputCount - i)]);
					}
					agent.Calc(inputs);

					//int maxIndex = 0;
					//float maxValue = 0f;
					//for(int i=0; i<agent.OutputLayer.PerceptCount; ++i) {
					//	if(agent.OutputLayer.percepts[i].net > maxValue) {
					//		maxValue = agent.OutputLayer.percepts[i].net;
					//		maxIndex = i;
					//	}
					//}
					//char result = (char)(maxIndex);
					char result = GetCharFromFloat(agent.OutputLayer.percepts[0].net);

					charList.Add(result);
				}
				return new string(charList.ToArray());
			}
		}
		private static float GetValueFromChar(char value) {
			return ((float)value) / MaxChar;
		}
		private static char GetCharFromFloat(float value) {
			return (char)Mathf.RoundToInt(value * MaxChar);
		}
	}
}
