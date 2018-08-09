from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from wtforms import Form, StringField, PasswordField, validators, TextAreaField, FormField, FieldList
from passlib.hash import sha256_crypt
from functools import wraps
import os
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil
from sklearn.externals import joblib   



def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()



app = Flask(__name__)
log = logging.getLogger('werkzeug')
#log.disabled = True
#app.logger.disabled = True
app.debug = True

# Index
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/close')
def close():
    shutdown_server()
    return render_template('close.html')

# About
@app.route('/about')
def about():
    return render_template('about.html')


# Register Form Class
class RegisterForm(Form):
    name = StringField('Name', [validators.required(),validators.Length(min=1, max=50)])
    username = StringField('Username', [validators.required(),validators.Length(min=4, max=25)])
    email = StringField('Email', [validators.required(),validators.Length(min=6, max=50)])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.EqualTo('confirm', message='Passwords do not match')
    ])
    confirm = PasswordField('Confirm Password')


# User Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if not os.path.exists('static/sql/users.csv'):
        if not os.path.exists('static/sql'):
            os.makedirs('static/sql')
        if request.method == 'POST' and form.validate():
            name = form.name.data
            email = form.email.data
            username = form.username.data
            password = sha256_crypt.encrypt(str(form.password.data))
            users = pd.DataFrame([[name,email,username,password]], columns=['name','email','username','password'])
            users.to_csv('static/sql/users.csv',index=False)
            flash('You are now registered and can log in', 'success')
            return redirect(url_for('login'))
        return render_template('register.html', form=form)
    else:
        if request.method == 'POST' and form.validate():
            name = form.name.data
            email = form.email.data
            username = form.username.data
            password = sha256_crypt.encrypt(str(form.password.data))      
            # Save new user to csv      
            users = pd.read_csv("static/sql/users.csv")
            if username in users['username'].values:
                error = 'Username has been used. Please choose a different username or log in.'
                return render_template('register.html', error=error)
            else: 
                users = users.append({'name':name,'email': email,'username': username,'password': password}, ignore_index=True)
                users.to_csv('static/sql/users.csv',index=False)
                flash('You are now registered and can log in', 'success')
                return redirect(url_for('login'))    
        return render_template('register.html', form=form)


# User login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if not os.path.exists('static/sql/users.csv'):
        flash('There is no registered user in database. Please register first.')
        return redirect(url_for('register')) 
    else:
        if request.method == 'POST':
            # Get Form Fields
            username = request.form['username']
            password_candidate = request.form['password']     
            # Get user by username
            users = pd.read_csv("static/sql/users.csv")   
            result = users[users['username']==username]
            if len(result) == 1:
                # Get stored hash
                password = users.loc[users['username']==username,'password'].to_string(index=False)
                # Compare Passwords
                if sha256_crypt.verify(password_candidate, password):
                    # Passed
                    session['logged_in'] = True
                    session['username'] = username
                    session['name'] = users.loc[users['username']==username,'name'].to_string(index=False)
    
                    flash('You are now logged in', 'success')
                    return redirect(url_for('dashboard'))
                else:
                    error = 'Invalid login. Please check your password.'
                    return render_template('login.html', error=error)
            else:
                error = 'Username not found'
                return render_template('login.html', error=error)
        return render_template('login.html')

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))

# Dashboard
@app.route('/dashboard')
@is_logged_in
def dashboard():
    if not os.path.exists('static/sql/models.csv'):
        if not os.path.exists('static/sql'):
            os.makedirs('static/sql')
        models = pd.DataFrame(columns=['id','modelname','modeltype','modelfolder','author','create_date'])
        models.to_csv('static/sql/models.csv',index=False)
        return redirect(url_for('dashboard'))
    else:
        models = pd.read_csv("static/sql/models.csv")
        if len(models) > 0:
            return render_template('dashboard.html', models=models)
        else:
            msg = 'No Models Found'
            return render_template('dashboard.html', msg=msg, models=models)

# Remove Model
@app.route('/remove_model/<string:id>')
@is_logged_in
def remove_model(id):
    # Delete model by id
    models = pd.read_csv("static/sql/models.csv")
    models = models[models['id']!=int(id)]
    if os.path.exists('static/models/' + str(id)):
        shutil.rmtree('static/models/' + str(id), ignore_errors=True)
    models.to_csv('static/sql/models.csv',index=False)
    flash('Model Removed', 'success')
    return redirect(url_for('dashboard'))

# Train model
@app.route('/train_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def train(id):
    # Get models
    models = pd.read_csv("static/sql/models.csv")   
    # Redirect according to model type:
    modeltype = models.loc[models['id']==int(id),'modeltype'].to_string(index=False)
    if modeltype == 'image':
        return redirect(url_for('train_image', id=id))
    if modeltype == 'text':
        return redirect(url_for('train_text', id=id))
    return redirect(url_for('dashboard'))

# Use model
@app.route('/use_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def use(id):  
    # Get models
    models = pd.read_csv("static/sql/models.csv")   
    # Redirect according to model type:
    modeltype = models.loc[models['id']==int(id),'modeltype'].to_string(index=False)
    if modeltype == 'image':
        return redirect(url_for('use_image', id=id))
    if modeltype == 'text':
        return redirect(url_for('use_text', id=id))
    return redirect(url_for('dashboard'))

# - - - Forms - - -
class LabelForm(Form):
    labelname = StringField('Label Name', [validators.required(), validators.Length(min=1, max=255)])
    labeldata = TextAreaField('Text Data, separated by line', [validators.required(),validators.Length(min=100)])

class CombinedForm(Form):
    modelname = StringField('Model Name', [validators.required(), validators.Length(min=1, max=255)])
    labels = FieldList(FormField(LabelForm), min_entries=1)

# Add text model
@app.route('/add_text_model', methods=['GET', 'POST'])
@is_logged_in
def add_text_model():
    form = CombinedForm(request.form)
    if request.method == 'POST':
        form = CombinedForm(request.form)
        modelname = form.modelname.data
        modeltype = "text" 
        # Get list of model from csv
        models = pd.read_csv("static/sql/models.csv")
        if len(models) > 0:
            model_id = max(models['id']) + 1
        else:
            model_id = 1
        # Save model to csv
        modelfolder = os.path.join('static/models', str(model_id), modelname)
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        models = models.append({'id':model_id,'modelname':modelname,'modeltype':modeltype,'modelfolder':modelfolder,'author':session['name'],'create_date':str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}, ignore_index=True)
        models.to_csv('static/sql/models.csv',index=False)
        labels = form.labels.data
        labels = pd.DataFrame(labels)
        filepath = modelfolder + '/train_data.csv'
        labels.to_csv(filepath,index=False)
        from scripts.train_text import train_text_classification
        model = train_text_classification(filepath)
        joblib.dump(model, modelfolder + '/text.model')
        flash('Your Model has been trained successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('train_text.html', form=form)

@app.route('/train_text_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def train_text(id):
    # Get model name
    models = pd.read_csv("static/sql/models.csv")   
    modelfolder = models.loc[models['id']==int(id),'modelfolder'].to_string(index=False)
    form = CombinedForm(request.form)
    form.modelname.data = models.loc[models['id']==int(id),'modelname'].to_string(index=False)
    if request.method == 'POST':
        form = CombinedForm(request.form)
        if os.path.exists('static/models/' + str(id)):
            shutil.rmtree('static/models/' + str(id), ignore_errors=True)
        modelname = form.modelname.data
        models = models[models['id']!=int(id)]
        modeltype = "text" 
        model_id = str(id)
        # Save model to csv
        modelfolder = os.path.join('static/models', str(model_id), modelname)
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        models = models.append({'id':model_id,'modelname':modelname,'modeltype':modeltype,'modelfolder':modelfolder,'author':session['name'],'create_date':str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}, ignore_index=True)
        models.to_csv('static/sql/models.csv',index=False)
        labels = form.labels.data
        labels = pd.DataFrame(labels)
        filepath = modelfolder + '/train_data.csv'
        labels.to_csv(filepath,index=False)
        from scripts.train_text import train_text_classification
        model = train_text_classification(filepath)
        joblib.dump(model, modelfolder + '/text.model')
        flash('Your Model has been retrained successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('train_text.html',form=form)

@app.route('/use_text_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def use_text(id):
    # Get model name
    models = pd.read_csv("static/sql/models.csv")
    model = models[models['id']==int(id)]
    modelfolder = models.loc[models['id']==int(id),'modelfolder'].to_string(index=False)
    form = CombinedForm(request.form)
    form.modelname.data = models.loc[models['id']==int(id),'modelname'].to_string(index=False)
    if request.method == 'POST':
        form = CombinedForm(request.form)
        labels = form.labels.data
        labels = pd.DataFrame(labels)
        filepath = modelfolder + '/test_data.csv'
        labels.to_csv(filepath,index=False)
        text_model = joblib.load(modelfolder + '/text.model')
        from scripts.use_text import use_text_classification
        result = use_text_classification(filepath, text_model)
        return render_template('report_text.html', model=model, result=result)
    return render_template('use_text.html',form=form)


    
# Add model Form Class
class ImageModelForm(Form):
    modelname = StringField('Model Name', [validators.required(), validators.Length(min=1, max=255)])
	
# Add image model
@app.route('/add_image_model', methods=['GET', 'POST'])
@is_logged_in
def add_image_model():
    form = ImageModelForm(request.form)
    if request.method == 'POST' and form.validate():
        form = ImageModelForm(request.form)
        modelname = form.modelname.data
        modeltype = "image" 
        # Get list of model from csv
        models = pd.read_csv("static/sql/models.csv")
        if len(models) > 0:
            model_id = max(models['id']) + 1
        else:
            model_id = 1
        # Save model to csv
        modelfolder = os.path.join('static/models', str(model_id), modelname, 'train_photos')
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        models = models.append({'id':model_id,'modelname':modelname,'modeltype':modeltype,'modelfolder':modelfolder,'author':session['name'],'create_date':str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}, ignore_index=True)
        for upload in request.files.getlist("fileList"):
            filename = upload.filename
            head, tail = os.path.split( filename )
            head2, tail2 = os.path.split(head)
            target_folder = os.path.join(modelfolder,tail2)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)  
            destination = os.path.join(target_folder, tail)
            upload.save(destination)
        # Train model
        trainmodelfolder, tail = os.path.split(modelfolder)
        commandline = 'python -m scripts.retrain --bottleneck_dir=' + trainmodelfolder + '/bottlenecks --how_many_training_steps=500 --model_dir=scripts/models  --summaries_dir=' + trainmodelfolder + '/training_summaries --output_graph=' + trainmodelfolder + '/retrained_graph.pb  --output_labels=' + trainmodelfolder + '/retrained_labels.txt --architecture="mobilenet_0.50_224" --image_dir=' + modelfolder        
        os.system(commandline)
        models.to_csv('static/sql/models.csv',index=False) 
        flash('Your Model has been trained successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('train_image.html', form=form)

@app.route('/train_image_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def train_image(id):
    # Get model name
    models = pd.read_csv("static/sql/models.csv")   
    modelfolder = models.loc[models['id']==int(id),'modelfolder'].to_string(index=False)
    form = ImageModelForm(request.form)
    form.modelname.data = models.loc[models['id']==int(id),'modelname'].to_string(index=False)
    if request.method == 'POST' and form.validate():
        form = ImageModelForm(request.form)
        if os.path.exists('static/models/' + str(id)):
            shutil.rmtree('static/models/' + str(id), ignore_errors=True)
        modelname = form.modelname.data
        models = models[models['id']!=int(id)]
        modeltype = "image" 
        model_id = str(id)
        # Save model to csv
        modelfolder = os.path.join('static/models', str(model_id), modelname, 'train_photos')
        if not os.path.exists(modelfolder):
            os.makedirs(modelfolder)
        models = models.append({'id':model_id,'modelname':modelname,'modeltype':modeltype,'modelfolder':modelfolder,'author':session['name'],'create_date':str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}, ignore_index=True)
        for upload in request.files.getlist("fileList"):
            filename = upload.filename
            head, tail = os.path.split( filename )
            head2, tail2 = os.path.split(head)
            target_folder = os.path.join(modelfolder,tail2)
            if not os.path.exists(target_folder):
                os.mkdir(target_folder)
            destination = os.path.join(target_folder, tail)
            upload.save(destination)
        # Train model
        trainmodelfolder, tail = os.path.split(modelfolder)
        commandline = 'python -m scripts.retrain --bottleneck_dir=' + trainmodelfolder + '/bottlenecks --how_many_training_steps=500 --model_dir=scripts/models  --summaries_dir=' + trainmodelfolder + '/training_summaries --output_graph=' + trainmodelfolder + '/retrained_graph.pb  --output_labels=' + trainmodelfolder + '/retrained_labels.txt --architecture="mobilenet_0.50_224" --image_dir=' + modelfolder        
        os.system(commandline)
        models.to_csv('static/sql/models.csv',index=False) 
        flash('Your Model has been retrained successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('train_image.html', form=form)

# Use model
@app.route('/use_image_<string:id>', methods=['GET', 'POST'])
@is_logged_in
def use_image(id):
    # Get model by id
    models = pd.read_csv("static/sql/models.csv")
    model = models[models['id']==int(id)]  
    modelfolder = models.loc[models['id']==int(id),'modelfolder'].to_string(index=False)
    form = ImageModelForm(request.form)
    form.modelname.data = models.loc[models['id']==int(id),'modelname'].to_string(index=False)
    if request.method == 'POST':
        modelname = models.loc[models['id'] ==  int(id),'modelname'].to_string(index=False)
        modelfolder = models.loc[models['id'] ==  int(id),'modelfolder'].to_string(index=False)
        modelfolder.replace('\\', '/')
        modelfolder, tail = os.path.split(modelfolder)
        imagefolder =  os.path.join(modelfolder, 'test_photos')  
        if os.path.exists(imagefolder):
            shutil.rmtree(imagefolder, ignore_errors=True)
        if not os.path.exists(imagefolder):
            os.makedirs(imagefolder)
        for upload in request.files.getlist("fileList"):
            filename = upload.filename
            print(filename)
            upload.save(os.path.join(imagefolder, filename)) 
        # Use model
        import scripts.label_image as label
        html_string = label.main(imagefolder, modelfolder, modelname)       
        return render_template('report_image.html', model=model, html_string=html_string)
    return render_template('use_image.html', form=form, id=id)


@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
	
if __name__ == '__main__':
    app.secret_key='secret123'  
    app.run()