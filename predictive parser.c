#include <stdio.h>
#include <string.h>

char stack[50];
int top = -1;

void push(char str[]) {
    for (int i = strlen(str) - 1; i >= 0; i--) {
        if (str[i] != 'e')
            stack[++top] = str[i];
    }
}

void pop() {
    if (top >= 0)
        top--;
}

void displayStack() {
    for (int i = 0; i <= top; i++)
        printf("%c", stack[i]);
}

void displayInput(char input[]) {
    printf("\t");
    for (int i = 0; input[i] != '\0'; i++)
        printf("%c", input[i]);
}

int main() {

    char input[50];

    char table[5][6][10] = {
        {"$", "$", "TH", "$", "TH", "$"},
        {"+TH", "$", "e", "e", "$", "e"},
        {"$", "$", "FU", "$", "FU", "$"},
        {"e", "*FU", "e", "e", "$", "e"},
        {"$", "$", "(E)", "$", "i", "$"}
    };

    char nonTerminals[] = {'E', 'H', 'T', 'U', 'F'};

    printf("Enter input string (append with $): ");
    scanf("%s", input);

    push("$");
    push("E");

    printf("\nStack\tInput\tOutput\n\n");

    int i = 0;

    while (top >= 0) {

        displayStack();
        displayInput(input);

        char st = stack[top];
        char in = input[i];

        if ((in >= 'a' && in <= 'z') ||
            (in >= 'A' && in <= 'Z'))
            in = 'i';

        if (st == in) {
            pop();
            i++;
            printf("\tPop\n");
            continue;
        }

        int row = -1, col = -1;

        for (int r = 0; r < 5; r++) {
            if (nonTerminals[r] == st) {
                row = r;
                break;
            }
        }

        if (in == '+') col = 0;
        else if (in == '*') col = 1;
        else if (in == '(') col = 2;
        else if (in == ')') col = 3;
        else if (in == 'i') col = 4;
        else if (in == '$') col = 5;

        if (row == -1 || col == -1 || strcmp(table[row][col], "$") == 0) {
            printf("\nGiven string is not accepted\n");
            return 0;
        }

        pop();

        push(table[row][col]);

        printf("\t%c -> %s\n", st, table[row][col]);
    }

    if (top == -1)
        printf("\nGiven string is accepted\n");
    else
        printf("\nGiven string is not accepted\n");

    return 0;
}
