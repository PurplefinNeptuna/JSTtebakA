using System;
using System.Collections.Generic;
using System.Linq;

namespace JST {

	[Serializable]
	public class NeuralNetwork {
		public int neuronIdx = 0;
		public int layerCount = 1;
		public double learningRate = 0f;
		public Func<double, double> defaultActivationFunction;
		public Func<double, double, double> defaultDoActivationFunction;
		public List<OutputNode> outputNodes = new List<OutputNode>();
		public List<InputNode> inputNodes = new List<InputNode>();
		public List<Neuron> allNeurons = new List<Neuron>();
		public List<Neuron> neuronTraverse = new List<Neuron>();
		public List<Neuron> inputNeurons = new List<Neuron>();

		public NeuralNetwork(int inputCount, int outputCount, double learnRate, Func<double, double> activation, Func<double, double, double> doActivation) {
			learningRate = learnRate;
			defaultActivationFunction = activation;
			defaultDoActivationFunction = doActivation;

			for (int i = 0; i < inputCount; i++) {
				inputNodes.Add(new InputNode());
			}

			for (int i = 0; i < outputCount; i++) {
				Neuron outNeuron = new Neuron(inputCount, defaultActivationFunction, defaultDoActivationFunction) {
					name = "N" + neuronIdx.ToString(),
						learningRate = learningRate,
						layer = 0
				};
				neuronIdx++;
				OutputNode outNode = new OutputNode(outNeuron);
				outputNodes.Add(outNode);
				outNeuron.AddOutputNode(outNode, 0);

				for (int j = 0; j < inputCount; j++) {
					outNeuron.SetInputNode(inputNodes[j], j);
				}
				allNeurons.Add(outNeuron);
				inputNeurons.Add(outNeuron);
			}

			ResetTheta();

			Traverse();
		}

		public void SetLearningRate(double learnRate) {
			learningRate = learnRate;
			foreach (var neuron in allNeurons) {
				neuron.learningRate = learningRate;
			}
		}

		public void SetActivationFunction(Func<double, double> act, Func<double, double, double> doAct) {
			defaultActivationFunction = act;
			defaultDoActivationFunction = doAct;
			foreach (var neuron in allNeurons) {
				neuron.activator = defaultActivationFunction;
				neuron.doActivator = defaultDoActivationFunction;
			}
			FeedForwardAndBackprop();
		}

		public void Traverse() {
			neuronTraverse.Clear();
			Queue<Neuron> bfsQueue = new Queue<Neuron>();
			for (int i = 0; i < allNeurons.Count; i++) {
				bool inputNeuron = false;
				Neuron nowNeuron = allNeurons[i];
				List<INode> nowChild = nowNeuron.inputNodes;
				for (int j = 0; j < nowChild.Count; j++) {
					if (nowChild[j].GetType() == typeof(InputNode)) {
						inputNeuron = true;
						break;
					}
				}
				if (inputNeuron) {
					allNeurons[i].traverseVisited = true;
					bfsQueue.Enqueue(allNeurons[i]);
				}
			}
			while (bfsQueue.Count > 0) {
				Neuron nodeNow = bfsQueue.Dequeue();
				neuronTraverse.Add(nodeNow);
				List<INode> nextNode = nodeNow.GetOutputNode();
				for (int i = 0; i < nextNode.Count; i++) {
					if (nextNode[i].GetType() == typeof(Neuron)) {
						Neuron nextNeuron = (Neuron)nextNode[i];
						if (!nextNeuron.traverseVisited) {
							nextNeuron.traverseVisited = true;
							bfsQueue.Enqueue(nextNeuron);
						}
					}
				}
			}
			foreach (var neuron in neuronTraverse) {
				neuron.traverseVisited = false;
			}
		}

		public void FeedForwardAndBackprop() {
			//Feed Forward
			for (int i = 0; i < neuronTraverse.Count; i++) {
				neuronTraverse[i].Recalculate();
			}
			for (int i = 0; i < outputNodes.Count; i++) {
				outputNodes[i].Recalculate();
			}
			//Back Propagate
			for (int i = neuronTraverse.Count - 1; i >= 0; i--) {
				neuronTraverse[i].RecalculateDoErrorDoNet();
			}
		}

		public Neuron AddNeuron(string name, Func<double, double> act, Func<double, double, double> doAct) {
			Neuron newNeuron = new Neuron(inputNodes.Count, act, doAct) {
				name = name,
					learningRate = learningRate,
					layer = layerCount
			};
			for (int i = 0; i < inputNodes.Count; i++) {
				newNeuron.SetInputNode(inputNodes[i], i);
			}
			allNeurons.Add(newNeuron);
			return newNeuron;
		}

		public void SetInput(in List<double> inputs) {
			for (int i = 0; i < Math.Min(inputs.Count, inputNodes.Count); i++) {
				inputNodes[i].value = inputs[i];
			}
			FeedForwardAndBackprop();
		}

		public void SetTarget(in List<int> targets) {
			for (int i = 0; i < Math.Min(targets.Count, outputNodes.Count); i++) {
				outputNodes[i].SetTarget(targets[i]);
			}
			FeedForwardAndBackprop();
		}

		public void Recalculate() {
			Traverse();
			FeedForwardAndBackprop();
		}

		public void Train() {
			for (int i = 0; i < neuronTraverse.Count; i++) {
				neuronTraverse[i].SGD();
			}
			FeedForwardAndBackprop();
		}

		public void ResetTheta() {
			foreach (var neuron in allNeurons) {
				neuron.ResetTheta();
			}
		}

		public List<int> GetPrediction() {
			List<int> ans = new List<int>();
			foreach (var outNode in outputNodes) {
				ans.Add(outNode.GetPrediction());
			}
			return ans;
		}

		public List<double> GetOutput() {
			List<double> ans = new List<double>();
			foreach (var outNode in outputNodes) {
				ans.Add(outNode.GetOutput());
			}
			return ans;
		}

		public List<double> GetError() {
			List<double> ans = new List<double>();
			foreach (var outNode in outputNodes) {
				ans.Add(outNode.GetError());
			}
			return ans;
		}

		public List<string> GetTraversePath() {
			List<string> ans = new List<string>();
			for (int i = 0; i < neuronTraverse.Count; i++) {
				ans.Add(neuronTraverse[i].name);
			}
			return ans;
		}

		public void AddLayer(int node, string activation = null) {
			Func<double, double> act;
			Func<double, double, double> doAct;
			List<Neuron> newInputNeuron = new List<Neuron>();
			if (activation.Equals("sigmoid")) {
				act = (x) => (1f / (1f + Math.Pow(Math.E, -x)));
				doAct = (x, y) => (y * (1f - y));
			} else if (activation.Equals("relu")) {
				act = (x) => (x > 0f?x : 0f);
				doAct = (x, y) => (x > 0f?1f : 0f);
			} else {
				act = defaultActivationFunction;
				doAct = defaultDoActivationFunction;
			}
			for (int i = 0; i < node; i++) {
				Neuron newNeuron = AddNeuron("N" + neuronIdx.ToString(), act, doAct);
				neuronIdx++;
				for (int j = 0; j < inputNeurons.Count; j++) {
					inputNeurons[j].SetInputSize(node);
					inputNeurons[j].SetInputNode(newNeuron, i);
					newNeuron.AddOutputNode(inputNeurons[j],i);
				}
				newInputNeuron.Add(newNeuron);
			}
			inputNeurons = new List<Neuron>(newInputNeuron);
			layerCount++;

			ResetTheta();
			Traverse();
		}
	}
}