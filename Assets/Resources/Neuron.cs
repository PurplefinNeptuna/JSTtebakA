using System;
using System.Collections.Generic;

namespace JST {

	[Serializable]
	public class Neuron : INode {
		public double bias, lastNet, lastOutput, doErrorDoOutput, doErrorDoNet, learningRate;
		public List<double> theta = new List<double>();
		public List<INode> inputNodes = new List<INode>();
		public List<ValueTuple<INode, int>> outputNodes = new List < (INode, int) > ();
		public Func<double, double> activator;
		public Func<double, double, double> doActivator;
		public bool traverseVisited;
		public string name;

		public Neuron(int iCount, Func<double, double> func, Func<double, double, double> dfunc) {
			traverseVisited = false;
			SetInputSize(iCount);
			activator = func;
			doActivator = dfunc;
		}

		public void SetTheta(int idx, double newTheta) {
			if (idx >= theta.Count)
				return;
			theta[idx] = newTheta;
		}

		public double GetTheta(int idx) {
			if (idx >= theta.Count)
				return 0f;
			return theta[idx];
		}

		public double GetOutput() {
			return lastOutput;
		}

		public void SetInputNode(INode input, int idx) {
			if (idx >= inputNodes.Count)
				return;
			inputNodes[idx] = input;
			Recalculate();
		}

		public void AddOutputNode(INode output, int inputPos) {
			RemoveOutputNode(output);
			outputNodes.Add((output, inputPos));
		}

		public void RemoveOutputNode(INode output) {
			int foundIdx = -1;
			for (int i = 0; i < outputNodes.Count; i++) {
				if (output == outputNodes[i].Item1) {
					foundIdx = i;
				}
			}
			if (foundIdx != -1) {
				outputNodes.RemoveAt(foundIdx);
			}
		}

		public List<INode> GetOutputNode() {
			List<INode> resNodes = new List<INode>();
			foreach (var nodeData in outputNodes) {
				resNodes.Add(nodeData.Item1);
			}
			return resNodes;
		}

		public double GetDoErrorForBackProp(int idx) {
			return doErrorDoNet * GetTheta(idx);
		}

		public void Recalculate() {
			lastNet = 0f;
			for (int i = 0; i < inputNodes.Count; i++) {
				if (inputNodes[i] != null) {
					double inputi = inputNodes[i].GetOutput();
					lastNet += inputi * theta[i];
				}
			}
			lastNet += bias;
			lastOutput = activator(lastNet);
		}

		public void RecalculateDoErrorDoNet() {
			doErrorDoOutput = 0.0;
			for (int i = 0; i < outputNodes.Count; i++) {
				INode outi = outputNodes[i].Item1;
				int idx = outputNodes[i].Item2;
				doErrorDoOutput += outi.GetDoErrorForBackProp(idx);
			}
			doErrorDoNet = doErrorDoOutput * doActivator(lastNet, lastOutput);
		}

		public void SGD() {
			for (int i = 0; i < theta.Count; i++) {
				double dtheta = doErrorDoNet * inputNodes[i].GetOutput();
				theta[i] -= learningRate * dtheta;
			}
			bias -= learningRate * doErrorDoNet;
		}

		public void SetInputSize(int size) {
			int thetaOldSize = theta.Count;
			if (size < thetaOldSize) {
				theta.RemoveRange(size, thetaOldSize - size);
				inputNodes.RemoveRange(size, thetaOldSize - size);
			} else if (size > thetaOldSize) {
				if (size > theta.Capacity)
					theta.Capacity = size;
				if (size > inputNodes.Capacity)
					inputNodes.Capacity = size;

				for (int i = 0; i < size - thetaOldSize; i++) {
					theta.Add(0f);
					inputNodes.Add(null);
				}
			}
		}
	}
}