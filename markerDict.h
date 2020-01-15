#ifndef MARKERDICT_H
#define MARKERDICT_H


#include <vector>
#include <string>
#include <map>


using namespace std;


class markerDict {
	public:
	markerDict();
	string getMarker(vector<int> candidate);
	private:
	map<string, vector<int> > name2marker;
	map<vector<int>, string> marker2name;
	vector<int> one, two, three, four, five, six, seven, eight, nine, zero;
	vector<int> A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z;
	vector<int> question, heart, arrow_u, arrow_l, arrow_r, dot, square, farm;
	vector<vector<int> > markers;
};


#endif
