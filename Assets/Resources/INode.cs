namespace JST {

	public interface INode {
		double GetOutput();
		double GetDoErrorForBackProp(int inputIndex);
	}

}