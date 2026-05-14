#include <stdio.h>
#include <string.h>
#include <ctype.h>

char op[100];
char val[100][20];

int top1 = -1, top2 = -1, temp = 1;

int prec(char ch)
{
    if (ch == '+' || ch == '-')
        return 1;

    if (ch == '*' || ch == '/')
        return 2;

    return 0;
}

void generate()
{
    char a[20], b[20], t[20];
    char opr = op[top1--];

    strcpy(b, val[top2--]);
    strcpy(a, val[top2--]);

    sprintf(t, "t%d", temp++);

    printf("%s = %s %c %s\n", t, a, opr, b);

    strcpy(val[++top2], t);
}

int main()
{
    char exp[100];

    printf("Enter Expression: ");
    scanf("%s", exp);

    for (int i = 0; exp[i] != '\0'; i++)
    {
        char ch = exp[i];

        if (isalnum(ch))
        {
            val[++top2][0] = ch;
            val[top2][1] = '\0';
        }

        else if (ch == '(')
        {
            op[++top1] = ch;
        }

        else if (ch == ')')
        {
            while (op[top1] != '(')
                generate();

            top1--;
        }

        else
        {
            while (top1 != -1 &&
                   prec(op[top1]) >= prec(ch))
            {
                generate();
            }

            op[++top1] = ch;
        }
    }

    while (top1 != -1)
        generate();

    return 0;
}
