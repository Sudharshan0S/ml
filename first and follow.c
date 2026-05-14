#include <stdio.h>
#include <string.h>
#include <ctype.h>

#define MAX 50

char input[MAX][50];
char productions[MAX][20];
char nonTerminals[MAX];

char first[MAX][MAX], follow[MAX][MAX];

int n, prodCount = 0, ntCount = 0;

void add(char *set, char val)
{
    for (int i = 0; set[i]; i++)
        if (set[i] == val)
            return;

    int len = strlen(set);

    set[len] = val;
    set[len + 1] = '\0';
}

int findIndex(char c)
{
    for (int i = 0; i < ntCount; i++)
        if (nonTerminals[i] == c)
            return i;

    return -1;
}

void splitProductions()
{
    for (int i = 0; i < n; i++)
    {
        char lhs = input[i][0];
        char rhs[50];

        strcpy(rhs, strchr(input[i], '=') + 1);

        char *token = strtok(rhs, "|");

        while (token != NULL)
        {
            productions[prodCount][0] = lhs;
            productions[prodCount][1] = '=';
            strcpy(&productions[prodCount][2], token);

            prodCount++;

            token = strtok(NULL, "|");
        }
    }
}

void computeFirst(char c, int idx)
{
    for (int i = 0; i < prodCount; i++)
    {
        if (productions[i][0] == c)
        {
            int j = 2;

            while (productions[i][j])
            {
                if (!isupper(productions[i][j]))
                {
                    add(first[idx], productions[i][j]);
                    break;
                }
                else
                {
                    int nextIdx = findIndex(productions[i][j]);

                    computeFirst(productions[i][j], nextIdx);

                    for (int k = 0; first[nextIdx][k]; k++)
                        add(first[idx], first[nextIdx][k]);

                    break;
                }
            }
        }
    }
}

void computeFollow(char c, int idx)
{
    if (idx == 0)
        add(follow[idx], '$');

    for (int i = 0; i < prodCount; i++)
    {
        for (int j = 2; productions[i][j]; j++)
        {
            if (productions[i][j] == c)
            {
                if (productions[i][j + 1])
                {
                    char next = productions[i][j + 1];

                    if (!isupper(next))
                    {
                        add(follow[idx], next);
                    }
                    else
                    {
                        int nextIdx = findIndex(next);

                        for (int k = 0; first[nextIdx][k]; k++)
                            add(follow[idx], first[nextIdx][k]);
                    }
                }
            }
        }
    }
}

int main()
{
    printf("Enter number of productions: ");
    scanf("%d", &n);

    printf("Enter productions:\n");

    for (int i = 0; i < n; i++)
        scanf("%s", input[i]);

    for (int i = 0; i < n; i++)
    {
        nonTerminals[ntCount++] = input[i][0];
    }

    splitProductions();

    for (int i = 0; i < ntCount; i++)
    {
        first[i][0] = '\0';
        follow[i][0] = '\0';
    }

    for (int i = 0; i < ntCount; i++)
        computeFirst(nonTerminals[i], i);

    for (int i = 0; i < ntCount; i++)
        computeFollow(nonTerminals[i], i);

    printf("\nFIRST Sets:\n");

    for (int i = 0; i < ntCount; i++)
    {
        printf("FIRST(%c) = { %s }\n",
               nonTerminals[i],
               first[i]);
    }

    printf("\nFOLLOW Sets:\n");

    for (int i = 0; i < ntCount; i++)
    {
        printf("FOLLOW(%c) = { %s }\n",
               nonTerminals[i],
               follow[i]);
    }

    return 0;
}
