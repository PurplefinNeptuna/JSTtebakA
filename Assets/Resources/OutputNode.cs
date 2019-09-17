using System;

namespace JST {

	[Serializable]
	public class OutputNode : INode {

		public Neuron outputNeuron;
		public double target = 0f, doErrorDoOut = 0f;

		public OutputNode(Neuron output) {
			outputNeuron = output;
		}

		public void SetTarget(int tar) {
			target = (double)tar;
			Recalculate();
		}

		public void Recalculate() {
			doErrorDoOut = GetOutput() - target;
		}

		public double GetError() {
			return 0.5f * doErrorDoOut * doErrorDoOut;
		}

		public double GetOutput() {
			if (outputNeuron != null)
				return outputNeuron.GetOutput();
			else
				return 0f;
		}

		public int GetPrediction() {
			double outN = GetOutput();
			int ansi = outN >= 0.5f?1 : 0;
			return ansi;
		}

		public double GetDoErrorForBackProp(int idx) {
			return doErrorDoOut;
		}
	}
}