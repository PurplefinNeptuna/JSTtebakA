using System;

namespace JST {

	[Serializable]
	public class InputNode: INode {
		public double value = 0f;

		public InputNode() {
			value = 0f;
		}

		public double GetOutput(){
			return value;
		}

		public double GetDoErrorForBackProp(int idx){
			return 0f;
		}
	}
}