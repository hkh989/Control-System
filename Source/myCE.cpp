#include "myCE.h"
#define _CRT_SECURE_NO_WARNINGS
void transf(int _deorder, int _numorder, double _den[], double _num[])
{
	printf("****************Controller Design Function****************");
	printf("\n");
	printf("Enter the order of denominator : ");
	scanf_s("%d", &_deorder);
	printf("\n");
	printf("Enter the order of numerator:");
	scanf_s("%d", &_numorder);
	printf("\n");
	printf("Enter the Components of denominator : ");
	for (int i = 0; i < _deorder + 1; i++)
	{
		scanf_s("%lf", &_den[i]);
		printf("%f", _den[i]);
	}
	printf("\n");
	printf("Enter the Components of numerator : ");
	for (int i = 0; i < _numorder + 1; i++)
	{
		scanf_s("%lf", &_num[i]);

	}
	for (int i = 0; i < _numorder+1; i++)
	{
		if (i == 0)
		{
			if (i != _numorder) {
				if (_num[i] == 1)
				{
					printf("s^%d", _numorder);
				}
				else
				{
					printf("%fs^%d", _num[i], _numorder);
				}

			}
			else
				printf("%f", _num[i]);
		}
		else
		{
			if (i == _numorder)
			{
				printf("+%f", _num[i]);
			}
			else if (_numorder - i == 1)
			{
				printf("+%fs", _num[i]);
			}
			else
				printf("+%fs^%d", _num[i], _numorder - i);
		}
	}
	printf("\n----------------------------------------\n");

	for (int i = 0; i < _deorder+1; i++)
	{
		if (i == 0)
		{
			if (i != _deorder) {
				if (_den[i] == 1)
				{
					printf("s^%d", _deorder);
				}
				else
				{
					printf("%fs^%d", _den[i], _deorder);// 
				}
			}
			else
				printf("%f", _den[i]);
		}
		else
		{
			if (i == _deorder)
			{
				printf("+%f", _den[i]);
			}
			else if (_deorder - i == 1)
			{
				printf("+%fs", _den[i]);
			}
			else
				printf("+%fs^%d", _den[i], _deorder - i);
		}
	}


}
