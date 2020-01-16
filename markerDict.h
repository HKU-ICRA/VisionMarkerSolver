#ifndef MARKERDICT_H
#define MARKERDICT_H


#include <vector>
#include <string>
#include <map>


using namespace std;


/*
	Class that contains the dictionary of markers used to
	identify each marker.
*/
class markerDict {

	public:
	/*
		Constructor initializes each of the marker with
		their respective dictionary.
	*/
	markerDict();

	/*
		Args:
			candidate : the image that could potentially be a marker. It should be a 7 x 7 Mat.
		Returns:
			A string with the name of the marker. "none" if it is not a marker.
	*/
	string getMarker(cv::Mat candidate);

	private:
	/*
		MAX_ERROR_BITS = Maximum amount of bits that could be wrong for each marker
		MIN_BORDER_SUM = Minimum amount of 1-bits each candidate marker should have
	*/
	int MAX_ERROR_BITS = 0;	// this should remain 0 for now since 8 and 3 has a one bit difference
	int MIN_BORDER_SUM = 45;
	
	/*
		Args:
			marker : a potential marker, correct marker should have a sum of >= MIN_BORDER_SUM
		Returns:
			true if >= MIN_BORDER_SUM, else false
	*/
	bool borderCheck(cv::Mat marker);

	vector<int> one, two, three, four, five, six, seven, eight, nine, zero;
	vector<int> A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z;
	vector<int> question, heart, arrow_u, arrow_l, arrow_r, dot, square, farm;
	cv::Mat one_mat, two_mat, three_mat, four_mat, five_mat, six_mat, seven_mat, eight_mat, nine_mat, zero_mat;
	cv::Mat A_mat, B_mat, C_mat, D_mat, E_mat, F_mat, G_mat, H_mat, I_mat, J_mat, K_mat, L_mat, M_mat, N_mat, O_mat, P_mat, Q_mat, R_mat, S_mat, T_mat, U_mat, V_mat, W_mat, X_mat, Y_mat, Z_mat;
	cv::Mat question_mat, heart_mat, arrow_u_mat, arrow_l_mat, arrow_r_mat, dot_mat, square_mat, farm_mat;
	vector<cv::Mat> markers;
	vector<string> names;
};


#endif
