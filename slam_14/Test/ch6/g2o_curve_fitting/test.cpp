//
// Created by nuc on 18-7-4.
//


#include <iostream>
using namespace std;

class AcctABC
{

public:
    AcctABC() {
        cout << "father " << endl;

    }

    virtual void ViewAccout() const = 0;

    virtual ~AcctABC() {}
};

class Brass:public AcctABC
{
public:
    Brass()
    {
        a = 0;

    }

    virtual void ViewAccout() const;
    int a;
    virtual ~Brass() {}

};

void Brass::ViewAccout() const
{
    cout << "I am a son" << endl;
}


int main()
{
    cout << "Just test " << endl;
    Brass* t = new Brass();
    return 0;
}