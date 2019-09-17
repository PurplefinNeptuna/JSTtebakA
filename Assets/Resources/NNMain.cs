using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using JST;
using UnityEngine;
using UnityEngine.UI;

public class NNMain : MonoBehaviour {
	private List<BitArray> dataTrain = new List<BitArray>();
	public NeuralNetwork perceptronModel;
	public NeuralNetwork backpropModel;

	public GameObject inputPanel;
	public List<Toggle> data = new List<Toggle>();
	public Toggle aState;
	public Text hasil;
	public InputField slpEpoch;
	public InputField mlpEpoch;
	public Text slpAcc;
	public Text mlpAcc;
	public Text logText;

	private void Awake() {
		Func<double, double> sigmoid = (x) => (1f / (1f + Math.Pow(Math.E, -x)));
		Func<double, double> linear = (x) => x;
		Func<double, double, double> derSigmoid = (x, y) => (y * (1f - y));
		Func<double, double, double> derLinear = (x, y) => 1f;

		perceptronModel = new NeuralNetwork(63, 1, 0.01f, linear, derLinear);

		//model masukin dari belakang :v
		backpropModel = new NeuralNetwork(63, 1, 0.01f, sigmoid, derSigmoid);
		backpropModel.AddLayer(2, "relu");
		backpropModel.AddLayer(4, "relu");
		backpropModel.AddLayer(8, "relu");
	}

	private void Start() {
		data = new List<Toggle>(inputPanel.GetComponentsInChildren<Toggle>());

		for (int i = 0; i < backpropModel.layerCount; i++) {
			Debug.Log(backpropModel.NeuronInLayer(i).Count);
		}
	}

	public void TambahData(){
		dataTrain.Add(GetRawData());
	}

	public void TebakSLP(){
		BitArray tebakData = GetRawData();
		List<int> ans = perceptronModel.GetPredictionFromInput(tebakData);
		if(ans[0]==tebakData.GetTargetData()[0]){
			hasil.text = "Huruf A";
		}
		else{
			hasil.text = "Bukan huruf A";
		}
	}

	public void TebakMLP(){
		BitArray tebakData = GetRawData();
		List<int> ans = backpropModel.GetPredictionFromInput(tebakData);
		if(ans[0]==tebakData.GetTargetData()[0]){
			hasil.text = "Huruf A";
		}
		else{
			hasil.text = "Bukan huruf A";
		}
	}

	public void TrainSLP(){
		perceptronModel.Fit(0.01f,int.Parse(slpEpoch.text),dataTrain);
		slpAcc.text = perceptronModel.Accuracy(dataTrain).ToString();
	}

	public void TrainMLP(){
		backpropModel.Fit(0.01f,int.Parse(mlpEpoch.text),dataTrain);
		mlpAcc.text = backpropModel.Accuracy(dataTrain).ToString();
	}

	public BitArray GetRawData() {
		List<bool> datum = new List<bool>();
		foreach (var dataT in data) {
			datum.Add(dataT.isOn);
		}
		datum.Add(aState.isOn);
		return new BitArray(datum.ToArray());
	}

	public string DataToString(BitArray dat) {
		byte[] datb = new byte[8];
		dat.CopyTo(datb, 0);
		return BitConverter.ToString(datb);
	}

}