using System.Collections;
using System.Collections.Generic;

namespace JST {
	public static class Trainer {
		public static void Fit(this NeuralNetwork nn, double lrate, int epoch, in List<BitArray> trainData) {
			nn.ResetTheta();
			nn.SetLearningRate(lrate);
			for (int i = 0; i < epoch; i++) {
				for (int j = 0; j < trainData.Count; j++) {
					nn.SetInput(trainData[j].GetInputData());
					nn.SetTarget(trainData[j].GetTargetData());
					nn.Train();
				}
			}
		}

		public static double Accuracy(this NeuralNetwork nn, in List<BitArray> checkData) {
			double total = 0;
			for (int i = 0; i < checkData.Count; i++) {
				nn.SetInput(checkData[i].GetInputData());
				if (nn.GetPrediction()[0] == checkData[i].GetTargetData()[0]) {
					total += 1f;
				}
			}
			return total / checkData.Count;
		}

		public static List<int> GetPredictionFromInput(this NeuralNetwork nn, BitArray input) {
			nn.SetInput(input.GetInputData(true));
			return nn.GetPrediction();
		}

		public static List<double> GetInputData(this BitArray datum, bool dirty = true) {
			List<double> ans = new List<double>();
			for (int i = 0; i < datum.Count - (dirty?1 : 0); i++) {
				ans.Add(datum[i] ? 1f : 0f);
			}
			return ans;
		}

		public static List<int> GetTargetData(this BitArray datum) {
			return new List<int> { datum[datum.Count - 1] ? 1 : 0 };
		}
	}
}