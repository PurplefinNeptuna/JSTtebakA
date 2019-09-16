using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using UnityEngine;

interface IName {
	string PrintName();
}

[Serializable]
public class Identifier : IName {
	public string name;
	public ValueTuple<int, int, int> ID;
	public Func<int, int, int, int> activator;

	public string PrintName() {
		StringBuilder sb = new StringBuilder();
		sb.Append(name);
		sb.AppendFormat(", {0:D} {1:D} {2:D} = {3:D}", ID.Item1, ID.Item2, ID.Item3, activator(ID.Item1, ID.Item2, ID.Item3));
		return sb.ToString();
	}
}

public class Tester : MonoBehaviour {

	private void Awake() {
		Identifier test = new Identifier {
			ID = (1, 2, 3),
				name = "eet",
				activator = (a, b, c) => a + b + c
		};

		IFormatter formatter = new BinaryFormatter();
		Stream stream = new FileStream("test.dat", FileMode.Create, FileAccess.Write);
		formatter.Serialize(stream, test);
		stream.Close();
	}

	private void Start() {
		IFormatter formatter = new BinaryFormatter();
		Stream stream = new FileStream("test.dat", FileMode.Open, FileAccess.Read);
		Identifier idf = (Identifier)formatter.Deserialize(stream);
		Debug.Log(idf.PrintName());
	}
}