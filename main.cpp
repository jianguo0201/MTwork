#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Python.h>
#include <arrayobject.h>
#include "./tensor/core/CHeader.h"
#include "./tensor/function/FHeader.h"
#include "./tensor/loss/LHeader.h"
#include "./network/XNet.h"
#include "./tensor/XGlobal.h"
#include "./tensor/XUtility.h"
#include "./tensor/XTensor.h"
#include "./sample/fnnlm/FNNLM.h"

namespace py = pybind11;
using namespace nts;
using namespace fnnlm;

#define MAX_NAME_LENGTH 1024

int init_numpy() {
	import_array(); // PyError if not successful
	return 0;
}

const static int numpy_initialized = init_numpy();

/*some functions for creating a new tensor from list*/
static int to_aten_dim_list(py::list list) {
	//compute the dim of list
	int dim = 1;
	for (auto item_list : list) {
		if (PySequence_Check(item_list.ptr())) {
			return dim + to_aten_dim_list(py::cast<py::list>(item_list));
		}
		else
			return dim;
	}
	return dim;
}

static int* to_aten_shape_list(int dim, py::list list) {
	//compute the shape of list
	int* shape;
	shape = (int*)malloc(sizeof(int) * dim);
	py::list item_list;
	if (dim != 1)
		item_list = py::cast<py::list>(list[0]);
	shape[0] = static_cast<int>(PyObject_Length(list.ptr()));
	for (int i = 1; i < dim - 1; ++i) {
		shape[i] = static_cast<int>(PyObject_Length(item_list.ptr()));
		item_list = py::cast<py::list>(item_list[0]);
	}
	if (dim != 1)
		shape[dim - 1] = static_cast<int>(PyObject_Length(item_list.ptr()));

	return shape;
}

XTensor tensor_from_list(py::list list) {
	//create a new tensor from list
	py::array_t<float> array = py::cast<py::array>(list);
	auto dim = to_aten_dim_list(list);
	auto shape = to_aten_shape_list(dim, list);
	XTensor floatTensor = NewTensor(dim, shape, X_FLOAT, -1);
	auto requestArray = array.request();
	void* data_ptr = requestArray.ptr;
	floatTensor.SetData(data_ptr, floatTensor.unitNum);
	return floatTensor;
}

//create a new tensor
XTensor Tensor(py::handle input) {
	PyObject* obj = input.ptr();
	if (PySequence_Check(obj)) {
		py::list list = py::cast<py::list>(input);
		return tensor_from_list(list);
	}
	else {
		return 0;
	}
}

//create a new empty tensor
XTensor TensorNew(int myOrder, py::list DimSize) {
	int * myDimSize = new int[myOrder];
	int i = 0;
	for (auto item : DimSize) {
		if (i < myOrder) {
			myDimSize[i] = py::cast<int>(item);
			i++;
		}
	}
	XTensor output = NewTensor(myOrder, myDimSize, X_FLOAT, -1);
	return output;
}

//create a new zero tensor
XTensor ZeroNewTensor(int myOrder, py::list DimSize) {
	int * myDimSize = new int [myOrder];
	int i = 0;
	for (auto item : DimSize) {
		if (i < myOrder) {
			myDimSize[i] = py::cast<int>(item);
			i++;
		}
	}
	XTensor output = NewTensor(myOrder, myDimSize, X_FLOAT, -1);
	output.SetZeroAll();
	return output;
}

//get the shape of tensor
py::list GetShape(XTensor tensor) {
	py::list shape;
	for (int i = 0; i < tensor.order; i++) {
		shape.append(tensor.dimSize[i]);
	}
	return shape;
}

//output format conversion
std::string  GetOutputFormat(std::vector<std::string> vecDate, int tOrder, int* tDimSzie, int digit) {
	std::string data = "";
	if (vecDate.empty())
		return data;

	if (tOrder == 1) {
		data += "[";
		for (int i = 0; i < vecDate.size() - 1; ++i) {
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";
				}
			}
			else
				data += vecDate.at(i);
			data += ", ";
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]";
		return data;
	}
	else if (tOrder == 2) {
		data += "[[";
		for (int i = 0; i < vecDate.size() - 1; ++i) {
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";
				}
			}
			else
				data += vecDate.at(i);
			if ((i + 1) % tDimSzie[1] == 0) {
				data += "],\n\t [";
			}
			else {
				data += ", ";
			}
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]]";
		return data;
	}
	else if (tOrder == 3) {
		data += "[[[";
		int countDim1 = 0;
		for (int i = 0; i < vecDate.size() - 1; i++) {
			if (vecDate.at(i).size() < digit) {
				data += vecDate.at(i);
				for (int j = 0; j < (digit - vecDate.at(i).size()); ++j) {
					data += "0";
				}
			}
			else
				data += vecDate.at(i);
			if (((i + 1) % tDimSzie[2] == 0) && ((countDim1 + 1) % tDimSzie[1] != 0)) {
				data += "],\n\t  [";
				countDim1++;
			}
			else if (((i + 1) % tDimSzie[2] == 0) && ((countDim1 + 1) % tDimSzie[1] == 0)) {
				data += "]],\n\n\t  [[";
				countDim1++;
			}
			else {
				data += ", ";
			}
		}
		if (vecDate.at(vecDate.size() - 1).size() < digit) {
			data += vecDate.at(vecDate.size() - 1);
			for (int j = 0; j < (digit - vecDate.at(vecDate.size() - 1).size()); ++j) {
				data += "0";
			}
		}
		else
			data += vecDate.at(vecDate.size() - 1);
		data += "]]]";
		return data;
	}
	else {
		for (int i = 0; i < vecDate.size() - 1; ++i) {
			data += vecDate.at(i);
			data += ", ";
		}
		data += vecDate.at(vecDate.size() - 1);
		return data;
	}
}

//get the data from tensor
std::string GetTensorData(XTensor tensor) {
	std::vector<std::string> vecDate;

	int digit = 0;
	int tOrder = tensor.order;
	int* tDimSize = new int[tOrder];
	for (int i = 0; i < tOrder; i++) {
		tDimSize[i] = tensor.GetDim(i);
	}
	if (tensor.dataType == X_DOUBLE)
		for (int i = 0; i < tensor.unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((double*)tensor.data)[i];
			if (buff.str().find('.') != -1) {
				vecDate.push_back(buff.str());
				if (digit < int(buff.str().size()))
					digit = int(buff.str().size());
			}
			else
				vecDate.push_back(buff.str() + '.');
		}
	else if (tensor.dataType == X_INT)
		for (int i = 0; i < tensor.unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((int*)tensor.data)[i];
			vecDate.push_back(buff.str());
		}
	else if (tensor.dataType == X_FLOAT)
		for (int i = 0; i < tensor.unitNum; ++i)
		{
			std::ostringstream buff;
			buff << ((float*)tensor.data)[i];
			if (buff.str().find('.') != -1) {
				if (digit < int(buff.str().size()))
					digit = int(buff.str().size());
				vecDate.push_back(buff.str());
			}
			else
				vecDate.push_back(buff.str() + '.');
		}
	else
		ShowNTErrors("TODO!");

	return GetOutputFormat(vecDate, tOrder, tDimSize, digit);
}

//get the item from XTensor by the index in the python
void throwError(int index, int length) {
	std::string errorString = "index ";
	std::ostringstream buff;
	buff << index;
	errorString += buff.str();
	errorString += " out of bounds for dimension 0 ";
	buff.str("");
	buff << (length - 1);
	errorString += "to ";
	errorString += buff.str();
	throw std::runtime_error(errorString);
}

XTensor GetDataFromTensorIndex(XTensor tensor, int index) {

	if (index >= tensor.GetDim(0)) {
		throwError(index, tensor.GetDim(0));
	}
	XTensor retTensor;
	DTYPE retData = tensor.Get1D(index);
	retTensor = NewTensor1D(1, tensor.dataType);
	retTensor.SetDataFixed(retData);
	return retTensor;
}

XTensor GetTensorFromTensorIndex(XTensor tensor, int index) {
	XTensor retTensor;
	if (index >= tensor.GetDim(0)) {
		throwError(index, tensor.GetDim(0));
	}

	XTensor srcIndex = NewTensor1D(1);
	XTensor tgtIndex = NewTensor1D(1);
	int srcIndexData[1] = { index };
	int tgtIndexData[1] = { 0 };
	srcIndex.SetData(srcIndexData, 1);
	tgtIndex.SetData(tgtIndexData, 1);
	retTensor = CopyIndexed(tensor, 0, srcIndex, tgtIndex, 1);
	SqueezeMe(retTensor);
	//DelTensor(srcIndex);
	//DelTensor(tgtIndex);
	//py::print("&&");
	return retTensor;
}

DTYPE TensorGet1D(XTensor tensor, int i) {

	return tensor.Get1D(i);
}

//some of arithmetic functions
XTensor TensorSum(const XTensor a, const XTensor b, DTYPE beta = (DTYPE) 1.0) {
	XTensor output;
	output = Sum(a, b, beta);
	return output;
}

XTensor TensorSub(const XTensor a, const XTensor b, DTYPE beta = (DTYPE) 1.0) {
	XTensor output;
	output = Sub(a, b, beta);
	return output;
}

XTensor TensorMultiply(const XTensor a, const XTensor b, int leadingDim = 0) {
	XTensor output;
	output = Multiply(a, b, leadingDim);
	return output;
}

XTensor TensorDiv(const XTensor a, const XTensor b, int leadingDim = 0) {
	XTensor output;
	output = Div(a, b, leadingDim);
	return output;
}

XTensor TensorMatrixMul(const XTensor a, const XTensor b, DTYPE alpha = (DTYPE) 1.0) {
	XTensor output;
	output = MatrixMul(a, b, alpha);
	return output;
}

XTensor TensorMulAndShift(const XTensor x, const XTensor w, const XTensor b, DTYPE alpha = 1.0) {
	XTensor output;
	output = MulAndShift(x, w, b, alpha);
	return output;

}

//some of math functions
XTensor TensorClip(const XTensor a, DTYPE lower, DTYPE upper) {
	XTensor output;
	output = Clip(a, lower, upper);
	return output;
}

XTensor TensorScaleAndShift(const XTensor a, DTYPE scale, DTYPE shift = 0) {
	XTensor output;
	output = ScaleAndShift(a, scale, shift);
	return output;
}

//some of movement functions
XTensor TensorCopyIndexed(const XTensor s, int dim,
	py::list srcIndex, py::list tgtIndex, int copyNum = 1) {
	XTensor output;
	XTensor src = Tensor(py::list(srcIndex));
	XTensor tgt = Tensor(py::list(tgtIndex));
	output = CopyIndexed(s, dim, src, tgt, copyNum);
	return output;
}

XTensor TensorGather(XTensor s, py::list(index)) {
	XTensor output;
	XTensor xindex = Tensor(index);
	output = Gather(s, xindex);
	return output;
}

XTensor TensorMerge(const XTensor &smallA, const XTensor &smallB, int whereToMerge) {
	XTensor output;
	output = Merge(smallA, smallB, whereToMerge);
	return output;
}

//some of reduce functions
XTensor TensorReduceMax(const XTensor input, int dim) {
	XTensor output;
	output = ReduceMax(input, dim);
	return output;
}

XTensor TensorReduceMean(const XTensor input, int dim) {
	XTensor output;
	output = ReduceMean(input, dim);
	return output;
}

XTensor TensorReduceSumAll(const XTensor source) {
	XTensor output;
	output = ReduceSumAll(source);
	return output;
}

DTYPE TensorReduceSumAllValue(const XTensor source) {
	DTYPE output;
	output = ReduceSumAllValue(source);
	return output;
}

//some functions from function
XTensor TensorHardTanH(const XTensor &x) {
	XTensor output;
	output = HardTanH(x);
	return output;
}

XTensor TensorSoftmax(const XTensor &x, int leadDim) {
	XTensor output;
	output = Softmax(x, leadDim);
	return output;
}

XTensor TensorRectify(const XTensor &x) {
	XTensor output;
	output = Rectify(x);
	return output;
}

XTensor TensorLogSoftmax(const XTensor &x, int leadDim) {
	XTensor output;
	output = LogSoftmax(x, leadDim);
	return output;
}

//set random data
XTensor TensorRand(int rNum, int cNum) {
	XTensor output;
	output.Rand(rNum, cNum);
	return output;
}

XTensor TensorRange(XTensor &tensor, DTYPE lower, DTYPE upper, DTYPE step) {
	tensor.Range(lower, upper, step);
	return tensor;
}

XTensor TensorSetDataRand(XTensor &tensor, DTYPE lower, DTYPE upper) {
	tensor.SetDataRand(lower, upper);
	return tensor;
}

XTensor TensorSetDataRandn(XTensor &tensor, DTYPE mean, DTYPE standardDeviation) {
	tensor.SetDataRandn(mean, standardDeviation);
	return tensor;
}

//function of fnnmodel
FNNModel Model() {
	return FNNModel();
}

//get the shape of hiddenW or hiddenB
py::list GetModelShape_hiddenW(FNNModel model) {
	py::list shape;
	shape = GetShape(model.hiddenW);
	return shape;
}

py::list GetModelShape_hiddenB(FNNModel model) {
	py::list shape;
	shape = GetShape(model.hiddenB);
	return shape;
}

void ModelCheck(FNNModel &model) {
	return Check(model);
}

void ModelInit(FNNModel &model) {
	return Init(model);
}

void ModelLoadArgs(FNNModel &model, py::list arglist) {
	int Arglist[5];
	int i = 0;
	for (auto item : arglist) {
		if (i < 5) {
			Arglist[i] = py::cast<int>(item);
			i++;
		}
	}
	model.n = Arglist[0];
	model.eSize = Arglist[1];
	model.vSize = Arglist[2];
	model.hDepth = Arglist[3];
	model.hSize = Arglist[4];
}

void ModelCopy(FNNModel &tgt, FNNModel &src) {
	return Copy(tgt, src);
}

void ModelClear(FNNModel &model, py::bool_ isNodeGrad) {
	return Clear(model, isNodeGrad);
}

void ModelDump(const char * fn, FNNModel &model) {
	return Dump(fn, model);
}

void ModelUpdate(FNNModel &model, FNNModel &grad, float epsilon, py::bool_ isNodeGrad) {
	return Update(model, grad, epsilon, isNodeGrad);
}

//Net
XNet Net() {
	return XNet();
}

void ModelForward(XTensor &input, XTensor &output, FNNModel &model) {
	int n = model.n;
	int depth = model.hDepth;
	XTensor hidden;
	hidden = input;
	/* hidden layers */
	for (int i = 0; i < depth; i++)
		hidden = HardTanH(MatrixMul(hidden, model.hiddenW[i]) + model.hiddenB[i]);

	/* output layer */
	output = Softmax(MatrixMul(hidden, model.outputW) + model.outputB, 1);
}

XTensor MakeGoldBatch(XTensor tensor, int itemNum, py::list cols) {
	int i = 0;
	XTensor output = tensor;
	/* set none-zero cells */
	for (auto item : cols) {
		if (i < itemNum) {
			output.Set2D(1.0F, i, py::cast<int>(item));
			i++;
		}
	}
	return output;
}

char* FileName(py::str train) {
	int lenth = len(train);
	char * trains = new char[lenth];
	int i = 0;
	for (auto item : train) {
		if (i < lenth) {
			trains[i] = py::cast<char>(item);
			i++;
		}
	}
	return trains;
}

void ModelTrain(py::bool_ isShuffled, FNNModel &model) {
	char * trains = new char[MAX_NAME_LENGTH];
	trains = "D:/lm-traindata/ptb/ptb.trainfix";
	return Train(trains, isShuffled, model);

}

double MGetClockSec() {
	return GetClockSec();
}

// compute the cross entropy loss
XTensor LossCrossEntropy(const XTensor & output, const XTensor & gold,
	int leadingDim = -1) {
	XTensor loss;
	loss = CrossEntropy(output, gold);
	return loss;
}

PYBIND11_MODULE(NiuTensor, m) {

	py::class_<XTensor>(m, "XTensor")
		//variable
		.def_readwrite("id", &XTensor::id)
		.def_readwrite("order", &XTensor::order)
		.def_readwrite("data", &XTensor::data)
		.def_readwrite("dataType", &XTensor::dataType)
		.def_readwrite("enableGrad", &XTensor::enableGrad)
		.def_readwrite("isGrad", &XTensor::isGrad)
		.def_readwrite("isVar", &XTensor::isVar)
		.def_readwrite("grad", &XTensor::grad)

		.def(py::init<>())
		.def("init", &XTensor::Init)
		.def("shape", &GetShape)

		//set data
		.def("Range", &TensorRange)
		.def("SetDataRand", &TensorSetDataRand)
		.def("SetDataRandn", &TensorSetDataRandn)

		.def("__getitem__",
			[](const XTensor& tensor, int index) {
		if (tensor.order == 1) {
			return GetDataFromTensorIndex(tensor, index);
		}
		else {
			return GetTensorFromTensorIndex(tensor, index);
		}
	})

		.def("__repr__",
			[](const XTensor& tensor) {
		return "tensor (" + GetTensorData(tensor) + ")";
	});

	py::class_<FNNModel>(m, "FNNModel")
		.def_readwrite("embeddingW", &FNNModel::embeddingW)
		.def_readwrite("outputW", &FNNModel::outputW)
		.def_readwrite("outputB", &FNNModel::outputB)
		.def_readwrite("n", &FNNModel::n)
		.def_readwrite("eSize", &FNNModel::eSize)
		.def_readwrite("hDepth", &FNNModel::hDepth)
		.def_readwrite("hSize", &FNNModel::hSize)
		.def_readwrite("vSize", &FNNModel::vSize)

		.def("hiddenWshape", &GetModelShape_hiddenW)
		.def("hiddenBshape", &GetModelShape_hiddenB);

	py::class_<XNet>(m, "XNet")
		.def_readwrite("id", &XNet::id)
		.def_readwrite("nodes", &XNet::nodes)
		.def_readwrite("gradNodes", &XNet::gradNodes)
		.def_readwrite("outputs", &XNet::outputs)
		.def_readwrite("inputs", &XNet::inputs)
		.def_readwrite("isGradEfficient", &XNet::isGradEfficient)

		.def("Backward", (void (XNet::*)(XTensor&)) &XNet::Backward);

	//management of tensor net (or graph)
	m.def("Net", &Net, py::return_value_policy::reference);
	m.def("GetClockSec", &MGetClockSec, py::return_value_policy::reference);
	
	//function of fnnmodel
	m.def("Model", &Model, py::return_value_policy::reference);
	m.def("LoadArgs", &ModelLoadArgs, py::return_value_policy::reference);
	m.def("Check", &ModelCheck, py::return_value_policy::reference);
	m.def("Init", &ModelInit, py::return_value_policy::reference);
	m.def("Copy", &ModelCopy, py::return_value_policy::reference);
	m.def("Dump", &ModelDump, py::return_value_policy::reference);
	m.def("Clear", &ModelClear, py::return_value_policy::reference);
	m.def("Update", &ModelUpdate, py::return_value_policy::reference);
	m.def("Forward", &ModelForward, py::return_value_policy::reference);
	m.def("Train", &ModelTrain, py::return_value_policy::reference);

	//function of tensor
	m.def("Tensor", &Tensor, py::return_value_policy::reference);
	m.def("NewTensor", &TensorNew, py::return_value_policy::reference);
	m.def("Zeros", &ZeroNewTensor, py::return_value_policy::reference);

	//set random data
	m.def("Rand", &TensorRand, py::return_value_policy::reference);
	m.def("Get1D", &TensorGet1D, py::return_value_policy::reference);
	m.def("MakeGoldBatch", &MakeGoldBatch, py::return_value_policy::reference);

	//some of arithmetic functions
	m.def("Sum", &TensorSum, py::arg("a"), py::arg("b"), py::arg("beta") = 1.0, py::return_value_policy::reference);
	m.def("Sub", &TensorSub, py::arg("a"), py::arg("b"), py::arg("beta") = 1.0, py::return_value_policy::reference);
	m.def("Multiply", &TensorMultiply, py::arg("a"), py::arg("b"), py::arg("leadingDim") = 0, py::return_value_policy::reference);
	m.def("Div", &TensorDiv, py::arg("a"), py::arg("b"), py::arg("leadingDim") = 0, py::return_value_policy::reference);
	m.def("MatrixMul", &TensorMatrixMul, py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0, py::return_value_policy::reference);
	m.def("MulAndShift", &TensorMulAndShift, py::arg("x"), py::arg("w"), py::arg("b"), py::arg("alpha") = 1.0, py::return_value_policy::reference);

	//some of math functions
	m.def("Clip", &TensorClip, py::return_value_policy::reference);
	m.def("ScaleAndShift", &TensorScaleAndShift, py::arg("a"), py::arg("scale"), py::arg("shift") = 0, py::return_value_policy::reference);

	//some of movement functions
	m.def("CopyIndexed", &TensorCopyIndexed, py::arg("s"), py::arg("dim"), py::arg("srcIndex"), py::arg("tgtIndex"), py::arg("copyNum") = 1, py::return_value_policy::reference);
	m.def("Gather", &TensorGather, py::return_value_policy::reference);
	m.def("Merge", &TensorMerge, py::return_value_policy::reference);

	//some of reduce functions
	m.def("ReduceMax", &TensorReduceMax, py::return_value_policy::reference);
	m.def("ReduceMean", &TensorReduceMean, py::return_value_policy::reference);
	m.def("ReduceSumAll", &TensorReduceSumAll, py::return_value_policy::reference);
	m.def("ReduceSumAllValue", &TensorReduceSumAllValue, py::return_value_policy::reference);

	//some functions from function
	m.def("HardTanH", &TensorHardTanH, py::return_value_policy::reference);
	m.def("Softmax", &TensorSoftmax, py::return_value_policy::reference);
	m.def("Rectify", &TensorRectify, py::return_value_policy::reference);
	m.def("LogSoftmax", &TensorLogSoftmax, py::return_value_policy::reference);
	
	// compute the cross entropy loss
	m.def("CrossEntropy", &LossCrossEntropy, py::arg("output"), py::arg("gold"), py::arg("leadingDim") = -1, py::return_value_policy::reference);

}
