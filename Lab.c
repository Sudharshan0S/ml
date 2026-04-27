#include<stdio.h>
#include<string.h>
#include<stdlib.h>

int i,top=-1;
void error(){
    printf("Syntax Error");
}
void push(char k[])
{
  for(i=0;k[i]!='\0';i++)
  {
    if(top<9)
      a[++top]=k[i];
  }
}
char TOS() {
  return a[top];
}
void pop() {
  if(top>=0)
    a[top--]='\0';
}

void display() {
  for(i=0;i<=top;i++)
    printf("%c",a[i]);
}

void display1(char p[],int m) {
  int l;
  printf("\t");
  for(l=m;p[l]!='\0';l++)
    printf("%c",p[l]);
}

char* stack(){
  return a;
}

char* strrev(char* str) {
    int len = strlen(str);
    int j;
    for (j = 0; j < len / 2; j++) {
        char temp = str[j];
        str[j] = str[len - j - 1];
        str[len - j - 1] = temp;
    }
    return str;
}
int main()
{
  char ip[20],r[20],st,an;
  int ir,ic,j=0;

  char t[5][6][10] = {
    {"$","$","TH","$","TH","$"},
    {"+TH","$","e","e","$","e"},
    {"$","$","FU","$","FU","$"},
    {"e","*FU","e","e","$","e"},
    {"$","$","(E)","$","i","$"}
  };

  printf("\nEnter any String(Append with $): ");

  fgets(ip, sizeof(ip), stdin);
  ip[strcspn(ip, "\n")] = '\0';

  printf("Stack\tInput\tOutput\n\n");

  push("$E");
  display();
  printf("\t%s\n",ip);

  for(j=0;ip[j]!='\0';)
  {
    an=ip[j];

    if(TOS()==an)
    {
      pop();
      display();
      display1(ip,j+1);
      printf("\tPOP\n");
      j++;
      continue;
    }
    st=TOS();

    if(st=='E') ir=0;
    else if(st=='H') ir=1;
    else if(st=='T') ir=2;
    else if(st=='U') ir=3;
    else if(st=='F') ir=4;
    else {
      error();
      break;
    }

    if(an=='+') ic=0;
    else if(an=='*') ic=1;
    else if(an=='(') ic=2;
    else if(an==')') ic=3;
    else if((an>='a'&&an<='z')||(an>='A'&&an<='Z')) { ic=4; an='i'; }
    else if(an=='$') ic=5;
    else {
      error();
      break;
    }

    strcpy(r, strrev(t[ir][ic]));
    strrev(t[ir][ic]);

    pop();
    push(r);

    if(TOS()=='e')
    {
      pop();
      display();
      display1(ip,j);
      printf("\t%c->e\n",st);
    }
    else {
      display();
      display1(ip,j);
      printf("\t%c->%s\n",st,t[ir][ic]);
    }

    if(TOS()=='$' && an=='$') {
      pop();
      j++;
      break;
    }

    if(TOS()=='$'){
      error();
      break;
    }
  }

  if(top == -1 && j == strlen(ip))
    printf("\n Given String is accepted");
  else
    printf("\n Given String is not accepted");

  return 0;
}
