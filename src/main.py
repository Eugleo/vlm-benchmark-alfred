import typer

app = typer.Typer()


@app.command()
def generate():
    activities = [Activity.from_file(p) for p in path.glob("**/traj_data.json")]
