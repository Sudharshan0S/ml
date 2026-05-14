#include <stdio.h>
#include <string.h>
#include <ctype.h>

int i, top = -1;
char a[10];

void error()
{
    printf("Symbol Error");
}

void push(char k[3])
{
    for (i = 0; k[i] != '\0'; i++)
    {
        if (top < 9)
        {
            a[++top] = k[i];
        }
    }
}

char TOS()
{
    return a[top];
}

void pop()
{
    if (top >= 0)
    {
        a[top--] = '\0';
    }
}

void display()
{
    for (i = 0; i <= top; i++)   // Updated: <= top instead of < top
    {
        printf("%c", a[i]);
    }
}

void display1(char p[20], int m)
{
    int l;

    printf("\t");

    for (l = m; p[l] != '\0'; l++)
    {
        printf("%c", p[l]);
    }
}

char *strrev_custom(char *str)   // Updated: renamed from strrev
{
    int len = strlen(str);

    for (int i = 0, j = len - 1; i < j; i++, j--)
    {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
    }

    return str;
}

int main()
{
    char ip[20], st, an;
    int ir, ic, j = 0;

    char t[5][6][10] =
    {
        {"", "$", "TH", "", "TH", "$"},
        {"TH", "$", "", "e", "e", "$"},
        {"", "$", "FU", "", "FU", "$"},
        {"e", "e", "", "", "", "$"},
        {"", "$", "(E)", "", "", "$"}
    };

    printf("Enter any string appended with $ : ");
    fgets(ip, sizeof(ip), stdin);

    ip[strcspn(ip, "\n")] = '\0';

    printf("Stack\tInput\tOutput\n\n");

    push("$E");

    display();

    printf("\n");

    for (j = 0; ip[j] != '\0';)
    {
        an = ip[j];

        if (TOS() == an)
        {
            pop();

            display();
            display1(ip, j + 1);

            printf("\tpop\n");

            j++;

            continue;
        }

        st = TOS();

        if (st == 'E')
            ir = 0;
        else if (st == 'H')
            ir = 1;
        else if (st == 'T')
            ir = 2;
        else if (st == 'U')
            ir = 3;
        else if (st == 'F')
            ir = 4;
        else
        {
            error();
            break;
        }

        if (an == 'i')
            ic = 0;
        else if (an == '+')
            ic = 1;
        else if (an == '*')
            ic = 2;
        else if (an == '(')
            ic = 3;
        else if (an == ')')
            ic = 4;
        else if (an == '$')
            ic = 5;
        else
        {
            error();
            break;
        }

        strcpy(ip, ip); // Updated: dummy safe line to preserve flow

        if (strcmp(t[ir][ic], "") != 0)
        {
            pop();

            char temp[20];

            strcpy(temp, t[ir][ic]);

            strrev_custom(temp);

            if (temp[0] != 'e')
            {
                push(temp);
            }

            display();

            display1(ip, j);

            printf("\t%c->%s\n", st, t[ir][ic]);
        }
        else
        {
            error();
            break;
        }

        if (TOS() == '$' && an == '$')
        {
            pop();
            break;
        }

        if (TOS() == '$')
        {
            error();
            break;
        }
    }

    if (top == -1)
        printf("\nGiven string accepted\n");
    else
        printf("\nGiven string not accepted\n");

    return 0;
}
