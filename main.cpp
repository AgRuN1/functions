#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <clocale>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <fstream>
#include <limits>

using namespace std;

#define for0(i, lim) for(int i = 0; i < lim; ++i)

typedef double real;
typedef vector<real> row;
typedef vector<row> matrix;
typedef pair<real, real> point;

#define x first
#define y second

const int INF = numeric_limits<int>::max();

void svd(const matrix& A, matrix& U, matrix& S, matrix& V);

int sign(real a);
real dot(const row& x, const row& y);
void round2zero(matrix& m, real l);
real diag_sum(const matrix& m);
matrix abs(const matrix& m);
void inverse(const matrix& A, matrix& A_);

matrix create(int width, int height);
matrix eye(int n);

matrix operator*(real a, const matrix& x);
matrix operator*(const matrix& a, const matrix& b);
matrix operator+(const matrix& a, const matrix& b);
matrix operator~(const matrix& m);

point operator*(real k, const point& p);
point operator+(const point& a, const point& b);
real operator*(const point& a, const point& b);
ostream& operator<<(ostream& out, const point& p);
real length(const point& p);

point grad(const point& p);
point descent(const point& x, const point& g, real step);
void shift(vector<point>& x, vector<point>& g, int n);
point diis(const matrix& A, vector<point>& p, int dim);

matrix r2m(const row& x);
row m2r(const matrix& m);

row column(const matrix& x, int k);
void rot(real f, real g, real& c, real& s);
matrix householder(row x, int k);

void read(matrix& A, row& B);
//типы узлов
enum NODE_TYPE {
	OPERATOR,
	CONSTANT,
	VARIABLE,
	FUNCTION
};
//типы переменных
enum VAR_TYPE {
	X, Y
};
//структура производной
struct Deriv {
	double df, val;
	bool depend;
};
//структура узла дерева
struct node {
	shared_ptr<node> left, right;
	NODE_TYPE type;
	virtual double value(double x, double y) = 0;
	virtual Deriv* diff(VAR_TYPE var, double x, double y) = 0;
};
//умный указатель
typedef shared_ptr<node> ptr;
//операторы
struct oper {
	unsigned priority;
	virtual double call(double a, double b) = 0;
	virtual Deriv* diff(Deriv* a, Deriv* b) = 0;
	oper* pset(unsigned p) {
		priority += 4 * p;
		return this;
	}
};
//сумма
struct sum : oper {
	sum() {
		priority = 1;
	}
	double call(double a, double b) {
		return a + b;
	}
	Deriv* diff(Deriv* a, Deriv* b) {
		Deriv* ans = new Deriv;
		ans->depend = a->depend || b->depend;
		ans->val = a->val + b->val;
		ans->df = (a->depend ? a->df : 0) + (b->depend ? b->df : 0);
		return ans;
	}
};
//умножение
struct mult : oper {
	mult() {
		priority = 2;
	}
	double call(double a, double b) {
		return a * b;
	}
	Deriv* diff(Deriv* a, Deriv* b) {
		Deriv* ans = new Deriv;
		ans->depend = a->depend || b->depend;
		ans->val = a->val * b->val;
		if (a->depend && b->depend) {
			ans->df = a->df * b->val + b->df * a->val;
		}
		else if (a->depend && !b->depend) {
			ans->df = a->df * b->val;
		}
		else if (!a->depend && b->depend) {
			ans->df = a->val * b->df;
		}
		else {
			ans->df = ans->val;
		}
		return ans;
	}
};
//возведение в степень
struct power : oper {
	power() {
		priority = 4;
	}
	double call(double a, double b) {
		return pow(a, b);
	}
	Deriv* diff(Deriv* a, Deriv* b) {
		Deriv* ans = new Deriv;
		ans->depend = a->depend || b->depend;
		ans->val = pow(a->val, b->val);
		if (a->depend && !b->depend) {
			ans->df = b->val * pow(a->val, b->val - 1) * a->df;
		}
		else if (!a->depend && b->depend) {
			ans->df = ans->val * log(a->val);
		}
		else {
			ans->df = ans->val;
		}
		return ans;
	}
};
//оператор
struct op_node : node {
	shared_ptr<oper> p;
	op_node(oper* p) : p(shared_ptr<oper>(p)) {
		type = OPERATOR;
	}
	double value(double x, double y) {
		return p->call(left->value(x, y), right->value(x, y));
	}
	Deriv* diff(VAR_TYPE var, double x, double y) {
		Deriv* la = left->diff(var, x, y);
		Deriv* ra = right->diff(var, x, y);
		Deriv* ans = p->diff(la, ra);
		delete la;
		delete ra;
		return ans;
	}
};
//переменная
struct var_node : node {
	VAR_TYPE var;
	var_node(VAR_TYPE var) : var(var)
	{
		type = VARIABLE;
	}
	double value(double x, double y) {
		return var == X ? x : y;
	}
	Deriv* diff(VAR_TYPE var, double x, double y) {
		Deriv* ans = new Deriv;
		if (var == this->var) ans->depend = 1;
		else ans->depend = 0;
		ans->val = (this->var == X ? x : y);
		ans->df = var == this->var;
		return ans;
	}
};
//константа
struct con_node : node {
	double val;
	con_node(double value) : val(value)
	{
		type = CONSTANT;
	}
	double value(double x, double y) {
		return val;
	}
	Deriv* diff(VAR_TYPE var, double x, double y) {
		Deriv* ans = new Deriv;
		ans->depend = 0;
		ans->val = val;
		ans->df = 0;
		return ans;
	}
};
// конфигуратор для DIIS
struct conf {
	double eps, alpha;
	int kmax;
};
istream& operator>>(istream& is, conf& c) {
	is >> c.eps >> c.alpha >> c.kmax;
	return is;
}
//класс функции
struct function {
	function(const string& s);
	virtual double call(double x, double y);
	virtual double call(point p) {
		return call(p.x, p.y);
	}
	virtual double get_diff(double val, double df) {
		return df;
	}
	virtual double get_val(double val) {
		return val;
	}
	point grad(const point &p);
	point min(const conf &c);
	// шаблонный метод для вычисления производной
	Deriv* difference(VAR_TYPE var, double x, double y) {
		Deriv* a = root->diff(var, x, y);
		Deriv* ans = new Deriv;
		ans->val = a->val;
		ans->depend = a->depend;
		if (a->depend) {
			ans->df = get_diff(a->val, a->df);
		}
		else {
			ans->df = get_val(ans->val);
		}
		delete a;
		return ans;
	}
	virtual double diff(VAR_TYPE var, double x, double y) {
		Deriv* ans = difference(var, x, y);
		double f = ans->depend ? ans->df : 0;
		delete ans;
		return f;
	}
protected:
	ptr root;
	unsigned build(int l, int r);
	vector<ptr> nodes;
	template <typename T>
	void add(T* a) {
		nodes.push_back(shared_ptr<T>(a));
	}
};
//узел вложенной функции
struct fun_node : node {
	unique_ptr <function> fun;
	fun_node(function* f) : fun(unique_ptr<function>(f)) {
		type = FUNCTION;
	}
	double value(double x, double y) {
		return fun->call(x, y);
	}
	Deriv* diff(VAR_TYPE var, double x, double y) {
		return fun->difference(var, x, y);
	}
};
//функции
//синус
struct sin_fun : function {
	using function::function;
	double call(double x, double y) {
		return sin(function::call(x, y));
	}
	double get_diff(double val, double df) {
		return cos(val) * df;
	}
	double get_val(double val) {
		return sin(val);
	}
};
//косинус
struct cos_fun : function {
	using function::function;
	double call(double x, double y) {
		return cos(function::call(x, y));
	}
	double get_diff(double val, double df) {
		return -sin(val) * df;
	}
	double get_val(double val) {
		return cos(val);
	}
};
//натуральный логарифм
struct ln_fun : function {
	using function::function;
	double call(double x, double y) {
		return log(function::call(x, y));
	}
	double get_diff(double val, double df) {
		return df / val;
	}
	double get_val(double val) {
		return log(val);
	}
};
// фабрика особых функций
function* create_fun(const string& name, const string& fun) {
	if (name == "sin")
		return new sin_fun(fun);
	if (name == "cos")
		return new cos_fun(fun);
	if (name == "ln")
		return new ln_fun(fun);
	return new function(fun);
}
// фабрика операторов
oper* create_oper(char c) {
	if (c == '+')
		return new sum;
	if (c == '*')
		return new mult;
	if (c == '^')
		return new power;
	throw "unknown operator"; // ошибка
}
double function::call(double x, double y) {
	return root->value(x, y);
}
//построение дерева
unsigned function::build(int l, int r) {
	if (l == r)
		return l;
	int minp = INF;
	vector<int> v;
	for (int i = l; i <= r; ++i) {
		if (nodes[i]->type == OPERATOR) {
			int p = static_cast<op_node*>(nodes[i].get())->p->priority;
			if (p == minp)
				v.push_back(i);
			else if (p < minp) {
				v.clear();
				v.push_back(i);
				minp = p;
			}
		}
	}
	int md = v[v.size() / 2];
	nodes[md]->left = nodes[build(l, md - 1)];
	nodes[md]->right = nodes[build(md + 1, r)];
	return md;
}
//Основной алгоритм
int main() {
	setlocale(LC_ALL, "RUS");
	freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
	string s;
	int n;
	conf c;
	cin >> n;
	for0(i, n) {
		cin >> s >> c;
		function f(s);
		point mn = f.min(c);
		cout << 'f' << mn << '=' << f.call(mn) << '\n';
	}
	return 0;
}
//конструктор функции
function::function(const string& s) {
	int state = 1;
	string digit;
	string func_name, func;
	unsigned brack_cur;
	unsigned brackets = 0;
	//операторы
	map<char, bool> mp;
	mp['*'] = 1;
	mp['+'] = 1;
	mp['^'] = 1;
	for (char c : s) {
		switch (state) {
		case 1: {
			if (c == 'x') {
				add(new var_node(X));
				state = 3;
			}
			else if (c == 'y') {
				add(new var_node(Y));
				state = 3;
			}
			else if (isdigit(c) || c == '.') {
				digit = c;
				state = 2;
			}
			else if (c == '-') {
				add(new con_node(-1));
				add(new op_node(create_oper('*')->pset(brackets)));
			}else if (c == '(') {
				++brackets;
			}
			else {
				func_name = c;
				state = 3;
			}
			break;
		}
		case 2: {
			if (c == '.' || isdigit(c)) {
				digit += c;
				break;
			}
			else if (c == ')') {
				--brackets;
				state = 3;
				add(new con_node(stof(digit)));
				break;
			}
			else {
				add(new con_node(stof(digit)));
			}
			state = 3;
		}
		case 3: {
			if (mp.count(c)) {
				add(new op_node(create_oper(c)->pset(brackets)));
				state = 1;
			}
			else if (c == '-') {
				add(new op_node(create_oper('+')->pset(brackets)));
				add(new con_node(-1));
				add(new op_node(create_oper('*')->pset(brackets)));
				state = 1;
			}
			else {
				func_name += c;
				state = 4;
			}
			break;
		}
		case 4: {
			if (c == '(') {
				brack_cur = brackets + 1;
				state = 5;
			}
			else {
				func_name += c;
			}
			break;
		}
		case 5: {
			if (c == '(')
				++brack_cur;
			if (c == ')')
				--brack_cur;
			if (brack_cur == brackets) {
				add(new fun_node(create_fun(func_name, func)));
				func = "";
				func_name = "";
				state = 3;
			}
			else {
				func += c;
			}
		}
		}
	}
	if (state == 2) {
		add(new con_node(stof(digit)));
	}
	root = nodes[build(0, nodes.size() - 1)];
}
//минимум функции
point function::min(const conf &c) {
	real eps = c.eps, step = c.alpha;
	int dim = 5, end = c.kmax, n = 1;
	matrix A = create(dim, dim);
	vector<point> x(dim);
	vector<point> g(dim);
	x[0] = { -0.5, -1 };
	g[0] = grad(x[0]);

	while (length(g[0]) >= eps) {
		shift(x, g, dim);
		x[0] = descent(x[1], g[1], step);
		g[0] = grad(x[0]);
		++n;
		if (n <= dim) {
			for0(i, n)
				for0(j, n)
				A[i][j] = g[i] * g[j];
			x[0] = diis(A, x, n);
		}
		else {
			for0(i, dim)
				for0(j, dim)
				A[i][j] = g[i] * g[j];
			x[0] = diis(A, x, dim);
		}
		g[0] = grad(x[0]);
		if (n == end) break;
	}
	if (n == end) cout << "Не сошлось\n";
	return x[0];
}
//вычисление градиента в точке
point function::grad(const point& p) {
	point g;
	g.x = diff(X, p.x, p.y);
	g.y = diff(Y, p.x, p.y);
	return g;
}
//DIIS
point diis(const matrix& A, vector<point>& x, int dim) {
	matrix A1 = create(dim, dim), A_;
	for0(i, dim)
		for0(j, dim)
		A1[i][j] = A[i][j];

	inverse(A1, A_);
	row c = m2r(A_ * ~r2m(row(dim, 1)));
	real s = 0;
	for0(i, dim) s += c[i];
	for0(i, dim) c[i] /= s;
	point r;
	for0(i, dim)
		r = r + c[i] * x[i];
	return r;
}
//Вычисление псевдообратной матрицы
void inverse(const matrix& A, matrix& A_) {
	matrix u, s, v;
	svd(A, u, s, v);
	matrix r = ~s;
	for0(i, s.size())
		if (r[i][i])
			r[i][i] = 1 / r[i][i];

	A_ = v * r * ~u;
}
//Сингулярное разложение
void svd(const matrix& A, matrix& U, matrix& S, matrix& V) {
	int k = A.size();
	// Бидиагонализация
	matrix U1 = eye(k);
	matrix V1 = eye(k);
	matrix A2 = A;
	matrix H1, H2;
	for0(i, k) {
		H1 = householder(column(A2, i), i);
		A2 = H1 * A2;
		U1 = U1 * ~H1;
		if (i < k - 2) {
			H2 = householder(A2[i], i + 1);
			A2 = A2 * ~H2;
			V1 = V1 * ~H2;
		}
	}
	//QR скачки
	matrix U2 = eye(U1.size());
	matrix V2 = eye(V1.size());
	matrix U3, V3, Q;
	real e = numeric_limits<real>::max();
	real c, s;
	while (e > 1e-13) {
		U3 = eye(k);
		V3 = eye(k);
		for0(i, k - 1) {
			rot(A2[i][i], A2[i][i + 1], c, s);
			Q = eye(k);
			Q[i][i] = c;
			Q[i][i + 1] = s;
			Q[i + 1][i] = -s;
			Q[i + 1][i + 1] = c;
			A2 = A2 * ~Q;
			V3 = V3 * ~Q;
			round2zero(A2, 1e-13);
			rot(A2[i][i], A2[i + 1][i], c, s);
			Q = eye(k);
			Q[i][i] = c;
			Q[i][i + 1] = s;
			Q[i + 1][i] = -s;
			Q[i + 1][i + 1] = c;
			A2 = Q * A2;
			U3 = U3 * ~Q;
			round2zero(A2, 1e-13);
		}
		e = diag_sum(A2);
		U2 = U2 * U3;
		V2 = V2 * V3;
	}
	U = U1 * U2;
	V = V1 * V2;
	S = A2;
}
//преобразование строки к матрице
matrix r2m(const row& x) {
	return matrix(1, x);
}
//преобразование матрицы к строке
row m2r(const matrix& m) {
	row r;
	for0(i, m.size())
		for0(j, m[0].size())
		r.push_back(m[i][j]);
	return r;
}
//знак числа
int sign(real a) {
	return (a > 0 ? 1 : (a == 0 ? 0 : -1));
}
//взятие всех значений матрицы по модулю
matrix abs(const matrix& m) {
	int height = m.size(), width = m[0].size();
	matrix r = create(width, height);
	for0(i, height)
		for0(j, width)
		r[i][j] = abs(m[i][j]);
	return r;
}
//сумма модулей диагнальных элементов матрицы
real diag_sum(const matrix& m) {
	real s = 0;
	for0(i, m.size())
		if(i + 1 < m.size())
			s += abs(m[i][i + 1]);
	return s;
}
//округление к нулю
void round2zero(matrix& m, real l) {
	for0(i, m.size()) {
		for0(j, m[0].size()) {
			if (abs(m[i][j]) < l)
				m[i][j] = 0;
		}
	}
}
//Создаем матрицу width на height
matrix create(int width, int height) {
	return matrix(height, row(width));
}

//умножение числа на матрицу
matrix operator*(real a, const matrix& x) {
	int height = x.size();
	int width = x[0].size();
	matrix r = create(width, height);
	for0(i, height)
		for0(j, width)
		r[i][j] = a * x[i][j];
	return r;
}
//сложение матриц
matrix operator+(const matrix& a, const matrix& b) {
	int height = a.size(), width = a[0].size();
	matrix r = create(width, height);
	for0(i, height)
		for0(j, width)
		r[i][j] = a[i][j] + b[i][j];
	return r;
}
//перемножение матриц
matrix operator*(const matrix& a, const matrix& b) {
	int height1 = a.size(), width1 = a[0].size(), height2 = b.size(), width2 = b[0].size();
	matrix r = create(width2, height1);
	for0(i, height1) {
		for0(j, width2) {
			for0(k, width1) {
				r[i][j] += a[i][k] * b[k][j];
			}
		}
	}
	return r;
}
//скалярное произведение векторов
real dot(const row& x, const row& y) {
	real result = 0;
	for0(i, min(x.size(), y.size())) {
		result += x[i] * y[i];
	}
	return result;
}
//транспонирование матриц
matrix operator~(const matrix& m) {
	int height = m.size(), width = m[0].size();
	matrix r = create(height, width);
	for (int i = 0; i < height; ++i)
		for (int j = 0; j < width; ++j)
			r[j][i] = m[i][j];
	return r;
}
//умножение точки на число
point operator*(real k, const point& p) {
	return { k * p.x, k * p.y };
}
//сложение точек
point operator+(const point& a, const point& b) {
	return { a.x + b.x, a.y + b.y };
}
//Скалярное произведение векторов
real operator*(const point& a, const point& b) {
	return a.x * b.x + a.y * b.y;
}
//вывод точки
ostream& operator<<(ostream& out, const point& p) {
	out << '(' << p.x << ',' << p.y << ')';
	return out;
}
//длина вектора до точки
real length(const point& p) {
	return sqrt(p.x * p.x + p.y * p.y);
}
//Градиентный спуск
point descent(const point& x, const point& g, real step) {
	return x + -step * g;
}
//сдвиг
void shift(vector<point>& x, vector<point>& g, int n) {
	for0(i, n - 1) {
		x[n - i - 1] = x[n - i - 2];
		g[n - i - 1] = g[n - i - 2];
	}
}
//создание единичной матрицы размерности n
matrix eye(int n) {
	matrix r = create(n, n);
	for0(i, n) r[i][i] = 1;
	return r;
}
//матрица преобразования Хаусхолдера
matrix householder(row x, int k) {
	int n = x.size();
	row u(n);
	real sum = 0;
	for (int i = k; i < n; ++i)
		sum += x[i] * x[i];
	u[k] = x[k] + sign(x[k]) * sqrt(sum);
	for (int i = k + 1; i < n; ++i)
		u[i] = x[i];
	return eye(n) + -2 / dot(u, u) * (~r2m(u) * r2m(u));
}
//получить k-ый столбец из матрицы
row column(const matrix& x, int k) {
	row r;
	for0(i, x.size())
		r.push_back(x[i][k]);
	return r;
}

void rot(real f, real g, real& c, real& s) {
	real t, t1;
	if (f == 0) {
		c = 0;
		s = 1;
	}
	else if (abs(f) > abs(g)) {
		t = g / f;
		t1 = sqrt(1 + t * t);
		c = 1 / t1;
		s = t * c;
	}
	else {
		t = f / g;
		t1 = sqrt(1 + t * t);
		s = 1 / t1;
		c = t * s;
	}
}