from flask import Flask, render_template, request
from rag import RAG

rag = RAG()

app =Flask(__name__)

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')

@app.route('/query', methods=['GET', 'POST'])
def chatbot_query():
    if request.method == 'POST':
        query = request.form.get('query')
        response = rag.qa(query)
        answer = response['answer']
        similar_books =  response['similar_books']
        return render_template('index.html', query=query, answer=answer, similar_books=similar_books)

if __name__ == '__main__':
    app.run(debug=True)

